//! In-process guard against unbounded agent↔agent reply loops.
//!
//! Two agents owned by the same human are, by default, mutually allowed to
//! respond to each other (the inbound author gate accepts same-owner
//! "siblings"). Nothing else in the harness bounds how many times they may
//! reply back and forth: `ignore_self` only drops an agent's *own* events, and
//! `max_turns_per_session` rotates a session for context hygiene rather than
//! capping a reply chain. So a pair of siblings that mention each other — or a
//! persona configured to respond to every message — can ping-pong forever.
//!
//! [`LoopGuard`] closes that gap with a small, in-process counter. For each
//! conversation it tracks how many consecutive *sibling-authored* events this
//! agent has responded to without an intervening human message. Once the count
//! exceeds the configured ceiling the agent stops auto-responding in that
//! conversation until a human (owner, allowlisted, or external) speaks again,
//! which resets the counter.
//!
//! The guard is deliberately local and allocation-cheap: no relay queries, no
//! cross-process coordination. Each agent process counts independently, so a
//! ping-pong is broken from both ends once either side hits the ceiling.

use std::collections::HashMap;
use std::sync::Mutex;

use uuid::Uuid;

/// Upper bound on the number of distinct conversations tracked at once. When
/// exceeded, the table is cleared wholesale (mirrors `OwnerCache`'s bounded
/// map). Clearing can only *reset* live counters — the safe direction, since a
/// reset re-grants replies rather than suppressing them incorrectly.
const MAX_TRACKED_CONVERSATIONS: usize = 1024;

/// Outcome of recording a triggering event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopDecision {
    /// The agent may respond to this event.
    Allow,
    /// The agent-to-agent reply chain exceeded the ceiling; drop this event.
    /// `chain` is the observed consecutive sibling-reply count.
    Suppress { chain: u32 },
}

/// Per-conversation counter of consecutive sibling-authored turns.
pub(crate) struct LoopGuard {
    counts: Mutex<HashMap<String, u32>>,
}

impl LoopGuard {
    pub(crate) fn new() -> Self {
        Self {
            counts: Mutex::new(HashMap::new()),
        }
    }

    /// Conversation key for `(channel, thread)`.
    ///
    /// Threaded replies share a stable root, so an agent↔agent ping-pong (which
    /// is threaded — agent replies are intentionally not flattened) accumulates
    /// under one key. Top-level (non-threaded) events share a single per-channel
    /// bucket so a persona that responds to *every* message is still bounded;
    /// the trade-off is that unrelated top-level chatter shares a counter, which
    /// is safe because any human message resets it.
    pub(crate) fn conversation_key(channel_id: Uuid, thread_root: Option<&str>) -> String {
        match thread_root {
            Some(root) => format!("{channel_id}:{root}"),
            None => format!("{channel_id}:_toplevel"),
        }
    }

    /// Record a triggering event and decide whether the agent may respond.
    ///
    /// - `author_is_agent` — true when the author is a same-owner sibling agent.
    ///   A human message (`false`) resets the conversation and always allows.
    /// - `max_chain` — ceiling on consecutive sibling replies; `0` disables the
    ///   guard entirely (always [`LoopDecision::Allow`]).
    ///
    /// The first `max_chain` sibling-authored events in a conversation are
    /// allowed; the `max_chain + 1`-th and beyond are suppressed until a human
    /// speaks.
    pub(crate) fn evaluate(
        &self,
        key: &str,
        author_is_agent: bool,
        max_chain: u32,
    ) -> LoopDecision {
        if max_chain == 0 {
            return LoopDecision::Allow;
        }
        // A poisoned lock must not wedge dispatch: fail open (Allow).
        let mut map = match self.counts.lock() {
            Ok(map) => map,
            Err(_) => return LoopDecision::Allow,
        };

        if !author_is_agent {
            // Human turn — the conversation is no longer a closed agent loop.
            map.remove(key);
            return LoopDecision::Allow;
        }

        // Bound memory before inserting a new key.
        if map.len() >= MAX_TRACKED_CONVERSATIONS && !map.contains_key(key) {
            map.clear();
        }

        let count = map.entry(key.to_string()).or_insert(0);
        *count = count.saturating_add(1);
        if *count > max_chain {
            LoopDecision::Suppress { chain: *count }
        } else {
            LoopDecision::Allow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chan() -> Uuid {
        Uuid::nil()
    }

    #[test]
    fn disabled_when_max_is_zero() {
        let g = LoopGuard::new();
        let key = LoopGuard::conversation_key(chan(), Some("root"));
        for _ in 0..100 {
            assert_eq!(g.evaluate(&key, true, 0), LoopDecision::Allow);
        }
    }

    #[test]
    fn allows_up_to_ceiling_then_suppresses() {
        let g = LoopGuard::new();
        let key = LoopGuard::conversation_key(chan(), Some("root"));
        // First `max_chain` sibling turns are allowed.
        for _ in 0..3 {
            assert_eq!(g.evaluate(&key, true, 3), LoopDecision::Allow);
        }
        // The next one crosses the ceiling.
        assert_eq!(
            g.evaluate(&key, true, 3),
            LoopDecision::Suppress { chain: 4 }
        );
        assert_eq!(
            g.evaluate(&key, true, 3),
            LoopDecision::Suppress { chain: 5 }
        );
    }

    #[test]
    fn human_message_resets_the_chain() {
        let g = LoopGuard::new();
        let key = LoopGuard::conversation_key(chan(), Some("root"));
        for _ in 0..3 {
            g.evaluate(&key, true, 3);
        }
        assert_eq!(
            g.evaluate(&key, true, 3),
            LoopDecision::Suppress { chain: 4 }
        );
        // A human speaks — chain resets, agent may respond again.
        assert_eq!(g.evaluate(&key, false, 3), LoopDecision::Allow);
        assert_eq!(g.evaluate(&key, true, 3), LoopDecision::Allow);
    }

    #[test]
    fn conversations_are_tracked_independently() {
        let g = LoopGuard::new();
        let a = LoopGuard::conversation_key(chan(), Some("root-a"));
        let b = LoopGuard::conversation_key(chan(), Some("root-b"));
        for _ in 0..4 {
            g.evaluate(&a, true, 3);
        }
        // `a` is suppressed, but `b` is untouched.
        assert!(matches!(
            g.evaluate(&a, true, 3),
            LoopDecision::Suppress { .. }
        ));
        assert_eq!(g.evaluate(&b, true, 3), LoopDecision::Allow);
    }

    #[test]
    fn top_level_events_share_a_per_channel_bucket() {
        let with_root = LoopGuard::conversation_key(chan(), Some("root"));
        let top_a = LoopGuard::conversation_key(chan(), None);
        let top_b = LoopGuard::conversation_key(chan(), None);
        assert_eq!(top_a, top_b);
        assert_ne!(with_root, top_a);
    }
}

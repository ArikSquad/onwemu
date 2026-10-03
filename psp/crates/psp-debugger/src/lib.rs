//! Structured diagnostics shared by the emulator and its frontend.
//!
//! Diagnostics are kept as data instead of being printed inside low-level
//! crates. The frontend can log them, show them in a debugger, or include them
//! in a test report without changing the subsystem that produced the event.

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
/// One guest-visible or host-side diagnostic event.
pub struct Diagnostic {
    /// Subsystem that produced the event, such as `cpu` or `gpu`.
    pub subsystem: &'static str,
    /// Stable event category used by logs and frontend filters.
    pub kind: &'static str,
    /// Human-readable explanation of what happened.
    pub message: String,
    /// Guest PC associated with the event, when one exists.
    pub pc: Option<u32>,
    /// Guest thread associated with the event, when one exists.
    pub thread: Option<u32>,
}
#[derive(Default)]
/// An append-only collection of diagnostics for one emulator run.
pub struct Diagnostics {
    events: Vec<Diagnostic>,
}
impl Diagnostics {
    /// Append an event while preserving arrival order.
    pub fn push(&mut self, event: Diagnostic) {
        self.events.push(event)
    }
    /// Borrow all events collected so far without clearing them.
    pub fn events(&self) -> &[Diagnostic] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_preserve_order_and_borrow_without_consuming() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.push(Diagnostic {
            subsystem: "cpu",
            kind: "syscall",
            message: "unknown syscall".into(),
            pc: Some(0x1000),
            thread: Some(4),
        });
        diagnostics.push(Diagnostic {
            subsystem: "gpu",
            kind: "signal",
            message: "display list paused".into(),
            pc: None,
            thread: None,
        });

        assert_eq!(diagnostics.events().len(), 2);
        assert_eq!(diagnostics.events()[0].subsystem, "cpu");
        assert_eq!(diagnostics.events()[1].kind, "signal");
        assert_eq!(diagnostics.events().len(), 2);
    }
}

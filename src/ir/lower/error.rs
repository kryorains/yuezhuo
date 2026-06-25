#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerError {
    pub message: String,
}

impl LowerError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

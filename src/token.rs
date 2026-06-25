#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub byte: usize,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

impl Span {
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Const,
    Int,
    Float,
    Void,
    If,
    Else,
    While,
    Break,
    Continue,
    Return,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Assign,
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
    And,
    Or,
    Not,

    // Special characters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,

    // Symbols and Identifiers
    Ident(String),
    IntConst(i64),
    FloatConst(f32),
    /// Runtime library calls may take string literals (not a SysY type, but must be tokenized).
    StringLiteral(String),

    // End of file
    Eof,
}

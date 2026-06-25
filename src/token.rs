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

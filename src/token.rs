#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Const, Int, Float, Void,
    If, Else, While, Break, Continue, Return,

    // Operators
    Plus, Minus, Star, Slash, Percent,
    Assign, Eq, Neq, Lt, Gt, Leq, Geq,
    And, Or, Not,

    // Special characters
    LParen, RParen, LBrace, RBrace,
    LBracket, RBracket, Comma, Semicolon,

    // Symbols and Identifiers
    Ident(String),
    IntConst(i32),
    FloatConst(f32),

    // End of file
    Eof,
}

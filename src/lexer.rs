use crate::token::Token;

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        let ch = match self.peek() {
            Some(ch) => ch,
            None => return Token::Eof,
        };

        // symbols can start with a letter or underscore
        if ch.is_ascii_alphabetic() || ch == '_' {
            return self.lex_ident_or_keyword();
        }

        // numbers can start with digits or a dot for floats
        if ch.is_ascii_digit()
            || (ch == '.' && self.peek_next().is_some_and(|c| c.is_ascii_digit()))
        {
            return self.lex_number();
        }

        // 符号
        match ch {
            '+' => {
                self.advance();
                Token::Plus
            }
            '-' => {
                self.advance();
                Token::Minus
            }
            '*' => {
                self.advance();
                Token::Star
            }
            '/' => {
                self.advance();
                Token::Slash
            }
            '%' => {
                self.advance();
                Token::Percent
            }
            '=' => {
                self.advance();
                if self.match_char('=') {
                    Token::Eq
                } else {
                    Token::Assign
                }
            }
            '!' => {
                self.advance();
                if self.match_char('=') {
                    Token::Neq
                } else {
                    Token::Not
                }
            }
            '<' => {
                self.advance();
                if self.match_char('=') {
                    Token::Leq
                } else {
                    Token::Lt
                }
            }
            '>' => {
                self.advance();
                if self.match_char('=') {
                    Token::Geq
                } else {
                    Token::Gt
                }
            }
            '&' => {
                self.advance();
                if self.match_char('&') {
                    Token::And
                } else {
                    panic!("Single & Illegal")
                }
            }
            '|' => {
                self.advance();
                if self.match_char('|') {
                    Token::Or
                } else {
                    panic!("Single | Illegal")
                }
            }
            '(' => {
                self.advance();
                Token::LParen
            }
            ')' => {
                self.advance();
                Token::RParen
            }
            '{' => {
                self.advance();
                Token::LBrace
            }
            '}' => {
                self.advance();
                Token::RBrace
            }
            '[' => {
                self.advance();
                Token::LBracket
            }
            ']' => {
                self.advance();
                Token::RBracket
            }
            ',' => {
                self.advance();
                Token::Comma
            }
            ';' => {
                self.advance();
                Token::Semicolon
            }
            '"' => self.lex_string_literal(),
            _ => panic!("Unknown: {}", ch),
        }
    }

    fn lex_ident_or_keyword(&mut self) -> Token {
        let mut s = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match s.as_str() {
            "const" => Token::Const,
            "int" => Token::Int,
            "float" => Token::Float,
            "void" => Token::Void,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "break" => Token::Break,
            "continue" => Token::Continue,
            "return" => Token::Return,
            _ => Token::Ident(s),
        }
    }

    fn lex_number(&mut self) -> Token {
        let mut s = String::new();
        let mut is_float = false;

        // Integer part (optional if starts with '.')
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Fractional part
        if self.peek() == Some('.') {
            // Treat as float if '.' is present and followed by digit, or if we already have digits.
            if self.peek_next().is_some_and(|c| c.is_ascii_digit()) || !s.is_empty() {
                is_float = true;
                s.push('.');
                self.advance();
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() {
                        s.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        // Exponent part (optional): e.g. 1e3, 1.2E-3
        if self.peek().is_some_and(|c| c == 'e' || c == 'E') {
            // Only accept exponent if we already have some numeric content
            if !s.is_empty() {
                is_float = true;
                s.push(self.advance().unwrap()); // e/E
                if self.peek().is_some_and(|c| c == '+' || c == '-') {
                    s.push(self.advance().unwrap());
                }
                let mut has_exp_digits = false;
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() {
                        has_exp_digits = true;
                        s.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
                if !has_exp_digits {
                    panic!("Invalid float exponent");
                }
            }
        }

        if is_float {
            Token::FloatConst(s.parse::<f32>().unwrap())
        } else {
            Token::IntConst(s.parse::<i32>().unwrap())
        }
    }

    fn lex_string_literal(&mut self) -> Token {
        // Consume opening quote
        let opening = self.advance();
        debug_assert_eq!(opening, Some('"'));

        let mut out = String::new();
        while let Some(ch) = self.peek() {
            match ch {
                '"' => {
                    self.advance(); // closing quote
                    return Token::StringLiteral(out);
                }
                '\\' => {
                    self.advance(); // consume '\'
                    let esc = self
                        .peek()
                        .unwrap_or_else(|| panic!("Unterminated string escape"));
                    self.advance();
                    match esc {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        '\\' => out.push('\\'),
                        '"' => out.push('"'),
                        '0' => out.push('\0'),
                        _ => panic!("Unsupported escape: \\{}", esc),
                    }
                }
                _ => {
                    out.push(ch);
                    self.advance();
                }
            }
        }
        panic!("Unterminated string literal");
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek_next(&self) -> Option<char> {
        self.input.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Whitespace
            while let Some(ch) = self.peek() {
                if ch.is_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }

            // Line comment: //...
            if self.peek() == Some('/') && self.peek_next() == Some('/') {
                self.advance();
                self.advance();
                while let Some(ch) = self.peek() {
                    self.advance();
                    if ch == '\n' {
                        break;
                    }
                }
                continue;
            }

            // Block comment: /* ... */
            if self.peek() == Some('/') && self.peek_next() == Some('*') {
                self.advance();
                self.advance();
                loop {
                    match (self.peek(), self.peek_next()) {
                        (Some('*'), Some('/')) => {
                            self.advance();
                            self.advance();
                            break;
                        }
                        (Some(_), _) => {
                            self.advance();
                        }
                        (None, _) => panic!("Unterminated block comment"),
                    }
                }
                continue;
            }

            break;
        }
    }
}

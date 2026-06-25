use crate::token::{Position, Span, SpannedToken, Token};

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
    byte_pos: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
            byte_pos: 0,
            line: 1,
            column: 1,
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.next_spanned_token().token
    }

    pub fn next_spanned_token(&mut self) -> SpannedToken {
        self.skip_whitespace_and_comments();
        let start = self.position();
        let token = self.lex_token();
        let end = self.position();
        SpannedToken {
            token,
            span: Span::new(start, end),
        }
    }

    fn lex_token(&mut self) -> Token {
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
        if self.peek() == Some('0') && self.peek_next().is_some_and(|c| c == 'x' || c == 'X') {
            return self.lex_hex_number();
        }

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
            let radix = if s.len() > 1 && s.starts_with('0') {
                if !s.chars().all(|c| matches!(c, '0'..='7')) {
                    panic!("Invalid octal integer literal: {}", s);
                }
                8
            } else {
                10
            };
            Token::IntConst(i64::from_str_radix(&s, radix).unwrap())
        }
    }

    fn lex_hex_number(&mut self) -> Token {
        self.advance(); // 0
        self.advance(); // x/X

        let mut int_digits = String::new();
        while self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
            int_digits.push(self.advance().unwrap());
        }

        let mut frac_digits = String::new();
        let has_dot = if self.peek() == Some('.') {
            self.advance();
            while self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
                frac_digits.push(self.advance().unwrap());
            }
            true
        } else {
            false
        };

        if int_digits.is_empty() && frac_digits.is_empty() {
            panic!("Invalid hexadecimal literal");
        }

        if self.peek().is_some_and(|c| c == 'p' || c == 'P') {
            self.advance();
            let exponent = self.lex_signed_decimal_exponent("hexadecimal float");
            return Token::FloatConst(hex_float_value(&int_digits, &frac_digits, exponent));
        }

        if has_dot {
            panic!("Hexadecimal float literal requires a binary exponent");
        }

        Token::IntConst(i64::from_str_radix(&int_digits, 16).unwrap())
    }

    fn lex_signed_decimal_exponent(&mut self, context: &str) -> i32 {
        let mut s = String::new();
        if self.peek().is_some_and(|c| c == '+' || c == '-') {
            s.push(self.advance().unwrap());
        }

        let mut has_digits = false;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            has_digits = true;
            s.push(self.advance().unwrap());
        }

        if !has_digits {
            panic!("Invalid {} exponent", context);
        }

        s.parse::<i32>().unwrap()
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
        self.byte_pos += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }

    fn position(&self) -> Position {
        Position {
            byte: self.byte_pos,
            line: self.line,
            column: self.column,
        }
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

fn hex_float_value(int_digits: &str, frac_digits: &str, exponent: i32) -> f32 {
    let mut value = 0.0f64;

    for ch in int_digits.chars() {
        value = value * 16.0 + hex_digit_value(ch) as f64;
    }

    let mut place = 1.0 / 16.0;
    for ch in frac_digits.chars() {
        value += hex_digit_value(ch) as f64 * place;
        place /= 16.0;
    }

    (value * 2f64.powi(exponent)) as f32
}

fn hex_digit_value(ch: char) -> u8 {
    ch.to_digit(16)
        .unwrap_or_else(|| panic!("Invalid hexadecimal digit: {}", ch)) as u8
}

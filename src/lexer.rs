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
        self.skip_whitespace();

        let ch = match self.peek() {
            Some(ch) => ch,
            None => return Token::Eof,
        };

        // symbols can start with a letter or underscore
        if ch.is_ascii_alphabetic() || ch == '_' {
            return self.lex_ident_or_keyword();
        }

        // numbers can start with digits or a dot for floats
        if ch.is_ascii_digit() {
            return self.lex_number();
        }

        // 符号
        match ch {
            '+' => { self.advance(); Token::Plus },
            '-' => { self.advance(); Token::Minus },
            '*' => { self.advance(); Token::Star },
            '/' => { self.advance(); Token::Slash },
            '%' => { self.advance(); Token::Percent },
            '=' => {
                self.advance();
                if self.match_char('=') { Token::Eq } else { Token::Assign }
            },
            '!' => {
                self.advance();
                if self.match_char('=') { Token::Neq } else { Token::Not }
            },
            '<' => {
                self.advance();
                if self.match_char('=') { Token::Leq } else { Token::Lt }
            },
            '>' => {
                self.advance();
                if self.match_char('=') { Token::Geq } else { Token::Gt }
            },
            '&' => {
                self.advance();
                if self.match_char('&') { Token::And } else { panic!("Single & Illegal") }
            },
            '|' => {
                self.advance();
                if self.match_char('|') { Token::Or } else { panic!("Single | Illegal") }
            },
            '(' => { self.advance(); Token::LParen },
            ')' => { self.advance(); Token::RParen },
            '{' => { self.advance(); Token::LBrace },
            '}' => { self.advance(); Token::RBrace },
            '[' => { self.advance(); Token::LBracket },
            ']' => { self.advance(); Token::RBracket },
            ',' => { self.advance(); Token::Comma },
            ';' => { self.advance(); Token::Semicolon },
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

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                s.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                s.push(ch);
                self.advance();
                is_float = true;
            } else {
                break;
            }
        }

        if is_float {
            Token::FloatConst(s.parse::<f32>().unwrap())
        } else {
            Token::IntConst(s.parse::<i32>().unwrap())
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
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

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }
}

use crate::ast::{Expr, Func, Program, Stmt, Type};
use crate::lexer::Lexer;
use crate::token::Token;

pub struct Parser {
    lexer: Lexer,
    cur: Token,
}

impl Parser {
    pub fn new(source: &str) -> Self {
        let mut lexer = Lexer::new(source);
        let cur = lexer.next_token();
        Self { lexer, cur }
    }

    fn bump(&mut self) {
        self.cur = self.lexer.next_token();
    }

    fn expect(&mut self, t: Token) {
        if self.cur == t {
            self.bump();
        } else {
            panic!("Expected {:?}, got {:?}", t, self.cur);
        }
    }

    fn expect_ident(&mut self) -> String {
        match &self.cur {
            Token::Ident(s) => {
                let out = s.clone();
                self.bump();
                out
            }
            _ => panic!("Expected Ident, got {:?}", self.cur),
        }
    }

    fn parse_type(&mut self) -> Type {
        match self.cur {
            Token::Int => {
                self.bump();
                Type::Int
            }
            Token::Void => {
                self.bump();
                Type::Void
            }
            Token::Float => {
                self.bump();
                Type::Float
            }
            _ => panic!("Expected type, got {:?}", self.cur),
        }
    }

    pub fn parse_program(&mut self) -> Program {
        let mut funcs = Vec::new();
        while self.cur != Token::Eof {
            funcs.push(self.parse_func_def());
        }
        Program { funcs }
    }

    fn parse_func_def(&mut self) -> Func {
        let ret = self.parse_type();
        let name = self.expect_ident();
        self.expect(Token::LParen);
        self.expect(Token::RParen);
        let body = self.parse_block();
        Func { name, ret, body }
    }

    fn parse_block(&mut self) -> Vec<Stmt> {
        self.expect(Token::LBrace);
        let mut stmts = Vec::new();
        while self.cur != Token::RBrace {
            stmts.push(self.parse_stmt());
        }
        self.expect(Token::RBrace);
        stmts
    }

    fn parse_stmt(&mut self) -> Stmt {
        match self.cur {
            Token::Return => {
                self.bump();
                let e = self.parse_expr();
                self.expect(Token::Semicolon);
                Stmt::Return(e)
            }
            _ => panic!("Only return stmt is supported for now, got {:?}", self.cur),
        }
    }

    fn parse_expr(&mut self) -> Expr {
        match self.cur {
            Token::IntConst(v) => {
                self.bump();
                Expr::Int(v)
            }
            _ => panic!(
                "Only int const expr is supported for now, got {:?}",
                self.cur
            ),
        }
    }
}

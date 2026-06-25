use crate::ast::{
    BinaryOp, Block, BlockItem, Decl, Def, Expr, Func, FuncParam, Init, Item, LValue, Program,
    Stmt, Type, UnaryOp,
};
use crate::lexer::Lexer;
use crate::token::{SpannedToken, Token};

pub struct Parser {
    lexer: Lexer,
    cur: SpannedToken,
    peek: SpannedToken,
}

impl Parser {
    pub fn new(source: &str) -> Self {
        let mut lexer = Lexer::new(source);
        let cur = lexer.next_spanned_token();
        let peek = lexer.next_spanned_token();
        Self { lexer, cur, peek }
    }

    fn bump(&mut self) {
        self.cur = self.peek.clone();
        self.peek = self.lexer.next_spanned_token();
    }

    fn expect(&mut self, t: Token) {
        if self.cur.token == t {
            self.bump();
        } else {
            self.error_at_current(&format!("Expected {:?}, got {:?}", t, self.cur.token));
        }
    }

    fn expect_ident(&mut self) -> String {
        match &self.cur.token {
            Token::Ident(s) => {
                let out = s.clone();
                self.bump();
                out
            }
            _ => self.error_at_current(&format!("Expected Ident, got {:?}", self.cur.token)),
        }
    }

    fn parse_type(&mut self) -> Type {
        match &self.cur.token {
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
            _ => self.error_at_current(&format!("Expected type, got {:?}", self.cur.token)),
        }
    }

    fn parse_btype(&mut self) -> Type {
        match &self.cur.token {
            Token::Int => {
                self.bump();
                Type::Int
            }
            Token::Float => {
                self.bump();
                Type::Float
            }
            _ => self.error_at_current(&format!("Expected base type, got {:?}", self.cur.token)),
        }
    }

    pub fn parse_program(&mut self) -> Program {
        let mut items = Vec::new();
        while self.cur.token != Token::Eof {
            items.push(self.parse_item());
        }
        Program { items }
    }

    fn parse_item(&mut self) -> Item {
        if self.cur.token == Token::Const {
            return Item::Decl(self.parse_decl());
        }

        let ty = self.parse_type();
        let name = self.expect_ident();
        if self.cur.token == Token::LParen {
            Item::Func(self.parse_func_def_after_name(ty, name))
        } else {
            Item::Decl(self.parse_var_decl_after_name(ty, name))
        }
    }

    fn parse_decl(&mut self) -> Decl {
        let is_const = self.cur.token == Token::Const;
        if is_const {
            self.bump();
        }

        let ty = self.parse_btype();
        let first = self.expect_ident();
        let mut defs = vec![self.parse_def_after_name(is_const, first)];

        while self.cur.token == Token::Comma {
            self.bump();
            let name = self.expect_ident();
            defs.push(self.parse_def_after_name(is_const, name));
        }

        self.expect(Token::Semicolon);
        Decl { is_const, ty, defs }
    }

    fn parse_var_decl_after_name(&mut self, ty: Type, name: String) -> Decl {
        let mut defs = vec![self.parse_def_after_name(false, name)];

        while self.cur.token == Token::Comma {
            self.bump();
            let name = self.expect_ident();
            defs.push(self.parse_def_after_name(false, name));
        }

        self.expect(Token::Semicolon);
        Decl {
            is_const: false,
            ty,
            defs,
        }
    }

    fn parse_def_after_name(&mut self, is_const: bool, name: String) -> Def {
        let dims = self.parse_array_dims();
        let init = if self.cur.token == Token::Assign {
            self.bump();
            Some(self.parse_init())
        } else if is_const {
            self.error_at_current("Const definition requires initializer");
        } else {
            None
        };

        Def { name, dims, init }
    }

    fn parse_array_dims(&mut self) -> Vec<Expr> {
        let mut dims = Vec::new();
        while self.cur.token == Token::LBracket {
            self.bump();
            dims.push(self.parse_expr());
            self.expect(Token::RBracket);
        }
        dims
    }

    fn parse_init(&mut self) -> Init {
        if self.cur.token == Token::LBrace {
            self.bump();
            let mut values = Vec::new();
            if self.cur.token != Token::RBrace {
                values.push(self.parse_init());
                while self.cur.token == Token::Comma {
                    self.bump();
                    values.push(self.parse_init());
                }
            }
            self.expect(Token::RBrace);
            Init::List(values)
        } else {
            Init::Expr(self.parse_expr())
        }
    }

    fn parse_func_def_after_name(&mut self, ret: Type, name: String) -> Func {
        self.expect(Token::LParen);
        let params = if self.cur.token == Token::RParen {
            Vec::new()
        } else {
            self.parse_func_params()
        };
        self.expect(Token::RParen);
        let body = self.parse_block();
        Func {
            name,
            ret,
            params,
            body,
        }
    }

    fn parse_func_params(&mut self) -> Vec<FuncParam> {
        let mut params = vec![self.parse_func_param()];
        while self.cur.token == Token::Comma {
            self.bump();
            params.push(self.parse_func_param());
        }
        params
    }

    fn parse_func_param(&mut self) -> FuncParam {
        let ty = self.parse_btype();
        let name = self.expect_ident();
        let mut dims = Vec::new();

        if self.cur.token == Token::LBracket {
            self.bump();
            self.expect(Token::RBracket);
            dims.push(None);

            while self.cur.token == Token::LBracket {
                self.bump();
                dims.push(Some(self.parse_expr()));
                self.expect(Token::RBracket);
            }
        }

        FuncParam { name, ty, dims }
    }

    fn parse_block(&mut self) -> Block {
        self.expect(Token::LBrace);
        let mut items = Vec::new();
        while self.cur.token != Token::RBrace {
            if self.cur.token == Token::Const
                || self.cur.token == Token::Int
                || self.cur.token == Token::Float
            {
                items.push(BlockItem::Decl(self.parse_decl()));
            } else {
                items.push(BlockItem::Stmt(self.parse_stmt()));
            }
        }
        self.expect(Token::RBrace);
        Block { items }
    }

    fn parse_stmt(&mut self) -> Stmt {
        match &self.cur.token {
            Token::LBrace => Stmt::Block(self.parse_block()),
            Token::If => self.parse_if_stmt(),
            Token::While => self.parse_while_stmt(),
            Token::Break => {
                self.bump();
                self.expect(Token::Semicolon);
                Stmt::Break
            }
            Token::Continue => {
                self.bump();
                self.expect(Token::Semicolon);
                Stmt::Continue
            }
            Token::Return => {
                self.bump();
                let value = if self.cur.token == Token::Semicolon {
                    None
                } else {
                    Some(self.parse_expr())
                };
                self.expect(Token::Semicolon);
                Stmt::Return(value)
            }
            Token::Semicolon => {
                self.bump();
                Stmt::Expr(None)
            }
            _ => self.parse_expr_or_assign_stmt(),
        }
    }

    fn parse_if_stmt(&mut self) -> Stmt {
        self.expect(Token::If);
        self.expect(Token::LParen);
        let cond = self.parse_expr();
        self.expect(Token::RParen);
        let then_branch = Box::new(self.parse_stmt());
        let else_branch = if self.cur.token == Token::Else {
            self.bump();
            Some(Box::new(self.parse_stmt()))
        } else {
            None
        };
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        }
    }

    fn parse_while_stmt(&mut self) -> Stmt {
        self.expect(Token::While);
        self.expect(Token::LParen);
        let cond = self.parse_expr();
        self.expect(Token::RParen);
        let body = Box::new(self.parse_stmt());
        Stmt::While { cond, body }
    }

    fn parse_expr_or_assign_stmt(&mut self) -> Stmt {
        let expr = self.parse_expr();
        if self.cur.token == Token::Assign {
            let Expr::LValue(target) = expr else {
                self.error_at_current("Assignment target must be an lvalue");
            };
            self.bump();
            let value = self.parse_expr();
            self.expect(Token::Semicolon);
            Stmt::Assign { target, value }
        } else {
            self.expect(Token::Semicolon);
            Stmt::Expr(Some(expr))
        }
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Expr {
        let mut expr = self.parse_logical_and();
        while self.cur.token == Token::Or {
            self.bump();
            expr = Expr::Binary {
                op: BinaryOp::Or,
                lhs: Box::new(expr),
                rhs: Box::new(self.parse_logical_and()),
            };
        }
        expr
    }

    fn parse_logical_and(&mut self) -> Expr {
        let mut expr = self.parse_equality();
        while self.cur.token == Token::And {
            self.bump();
            expr = Expr::Binary {
                op: BinaryOp::And,
                lhs: Box::new(expr),
                rhs: Box::new(self.parse_equality()),
            };
        }
        expr
    }

    fn parse_equality(&mut self) -> Expr {
        let mut expr = self.parse_relational();
        loop {
            let op = match self.cur.token {
                Token::Eq => BinaryOp::Eq,
                Token::Neq => BinaryOp::Ne,
                _ => break,
            };
            self.bump();
            expr = Expr::Binary {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(self.parse_relational()),
            };
        }
        expr
    }

    fn parse_relational(&mut self) -> Expr {
        let mut expr = self.parse_additive();
        loop {
            let op = match self.cur.token {
                Token::Lt => BinaryOp::Lt,
                Token::Gt => BinaryOp::Gt,
                Token::Leq => BinaryOp::Le,
                Token::Geq => BinaryOp::Ge,
                _ => break,
            };
            self.bump();
            expr = Expr::Binary {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(self.parse_additive()),
            };
        }
        expr
    }

    fn parse_additive(&mut self) -> Expr {
        let mut expr = self.parse_multiplicative();
        loop {
            let op = match self.cur.token {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.bump();
            expr = Expr::Binary {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(self.parse_multiplicative()),
            };
        }
        expr
    }

    fn parse_multiplicative(&mut self) -> Expr {
        let mut expr = self.parse_unary();
        loop {
            let op = match self.cur.token {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                Token::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.bump();
            expr = Expr::Binary {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(self.parse_unary()),
            };
        }
        expr
    }

    fn parse_unary(&mut self) -> Expr {
        match self.cur.token {
            Token::Plus => {
                self.bump();
                Expr::Unary {
                    op: UnaryOp::Pos,
                    expr: Box::new(self.parse_unary()),
                }
            }
            Token::Minus => {
                self.bump();
                Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(self.parse_unary()),
                }
            }
            Token::Not => {
                self.bump();
                Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(self.parse_unary()),
                }
            }
            _ => self.parse_primary_or_call(),
        }
    }

    fn parse_primary_or_call(&mut self) -> Expr {
        match &self.cur.token {
            Token::LParen => {
                self.bump();
                let expr = self.parse_expr();
                self.expect(Token::RParen);
                expr
            }
            Token::Ident(name) if self.peek.token == Token::LParen => {
                let name = name.clone();
                self.bump();
                self.expect(Token::LParen);
                let mut args = Vec::new();
                if self.cur.token != Token::RParen {
                    args.push(self.parse_expr());
                    while self.cur.token == Token::Comma {
                        self.bump();
                        args.push(self.parse_expr());
                    }
                }
                self.expect(Token::RParen);
                Expr::Call { name, args }
            }
            Token::Ident(_) => Expr::LValue(self.parse_lvalue()),
            Token::IntConst(v) => {
                let v = *v;
                self.bump();
                Expr::Int(v)
            }
            Token::FloatConst(v) => {
                let v = *v;
                self.bump();
                Expr::Float(v)
            }
            Token::StringLiteral(s) => {
                let s = s.clone();
                self.bump();
                Expr::String(s)
            }
            _ => self.error_at_current(&format!("Expected expression, got {:?}", self.cur.token)),
        }
    }

    fn parse_lvalue(&mut self) -> LValue {
        let name = self.expect_ident();
        let mut indices = Vec::new();
        while self.cur.token == Token::LBracket {
            self.bump();
            indices.push(self.parse_expr());
            self.expect(Token::RBracket);
        }
        LValue { name, indices }
    }

    fn error_at_current(&self, message: &str) -> ! {
        let pos = self.cur.span.start;
        panic!("{} at {}:{}", message, pos.line, pos.column);
    }
}

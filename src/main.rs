mod lexer;
mod token;

use lexer::Lexer;
use token::Token;

fn main() {
    let source = "const int a = 5;";
    let mut lexer = Lexer::new(source);

    println!("SysY: {}", source);
    println!("Tokenizer:");

    loop {
        let tok = lexer.next_token();
        println!("{:?}", tok);
        if tok == Token::Eof {
            break;
        }
    }
}

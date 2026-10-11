use std::io::{self, BufRead};

fn show(text: &str) {
    let p = text2color::predict(text);

    println!(
        "{:<30} {}  rgb({:>3},{:>3},{:>3})  confidence {:.2}",
        text,
        p.hex(),
        p.rgb[0],
        p.rgb[1],
        p.rgb[2],
        p.confidence
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        for line in io::stdin().lock().lines() {
            show(line.unwrap().trim());
        }
    } else {
        show(&args.join(" "));
    }
}

import json

with open('charlist.json', 'r') as input_file:
    with open('charlist.rs', 'w+') as output_file:
        content = json.dumps(json.loads(input_file.read()), ensure_ascii=False,
                             indent=None, separators=(',', ':'))
        output_file.write("""\
use lazy_static::lazy_static;
use std::collections::HashMap;
use serde::Deserialize;

pub type CharList = HashMap<char, HashMap<String, usize>>;

lazy_static! {{
	pub static ref CHARLIST: CharList = {{
        let charlist = r#"{content}"#;
        serde_json::from_slice(charlist).unwrap()
	}};
}}
""".format(content=content)
        )

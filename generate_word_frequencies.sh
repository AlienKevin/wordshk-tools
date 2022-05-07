#!/bin/bash

begin="use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
	pub static ref WORD_FREQUENCIES: HashMap<u32, u8> = {
		HashMap::from([
"
echo "$begin" > src/word_frequencies.rs
cat src/wordfreqvalues.txt | grep -v "^[0-9]+\|50$" | sed -E "s/([0-9]+)\|([0-9]+)/(\1,\2),/g" >> src/word_frequencies.rs
end="])
	};
}"
echo "$end" >> src/word_frequencies.rs

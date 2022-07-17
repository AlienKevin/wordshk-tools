common_simps = set()

with open("common_trad_to_simp.tsv") as file:
    for line in file:
        if line.startswith("#"):
            continue
        l = line.rstrip().split('\t')
        common_simps.add(l[1])

with open("simp_to_trad.tsv") as file:
    iconic_simps = []
    for line in file:
        if line.startswith("#"):
            continue
        l = line.rstrip().split('\t')
        simp = l[0]
        trads = l[1].split(' ')
        if not (simp in trads) and simp in common_simps:
            iconic_simps.append(simp)
        # output all iconic simplified characters
        output_file = open("iconic_simps.rs", "w")
        output_file.write("""\
use lazy_static::lazy_static;
use std::collections::HashSet;

lazy_static! {{
	pub static ref ICONIC_SIMPS: HashSet<char> = {{
		HashSet::from([
			{iconic_simps}
		])
	}};
}}
""".format(iconic_simps=",".join(map(lambda simp: "'" + simp + "'", iconic_simps)))
        )

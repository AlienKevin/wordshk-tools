common_simps = set()
common_trads = set()

with open("common_trad_to_simp.tsv") as file:
    for line in file:
        if line.startswith("#"):
            continue
        l = line.rstrip().split('\t')
        common_trads.add(l[0])
        common_simps.add(l[1])

hk_variants = set()

with open("hk_variants.tsv") as file:
    for line in file:
        if line.startswith("#"):
            continue
        l = line.rstrip().split('\t')
        hk_variants = hk_variants.union(set(l[1].split(' ')))

with open("simp_to_trad.tsv") as file:
    iconic_simps = []
    for line in file:
        if line.startswith("#"):
            continue
        l = line.rstrip().split('\t')
        simp = l[0]
        trads = l[1].split(' ')
        if not (simp in trads) and not (simp in hk_variants) and simp in common_simps:
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

with open("trad_to_simp.tsv") as file:
    iconic_trads = []
    for line in file:
        if line.startswith("#"):
            continue
        l = line.rstrip().split('\t')
        trad = l[0]
        simps = l[1].split(' ')
        if not (trad in simps) and not (simp in hk_variants) and trad in common_trads:
            iconic_trads.append(trad)

        # output all iconic traditional characters
        output_file = open("iconic_trads.rs", "w")
        output_file.write("""\
use lazy_static::lazy_static;
use std::collections::HashSet;

lazy_static! {{
	pub static ref ICONIC_TRADS: HashSet<char> = {{
		HashSet::from([
			{iconic_trads}
		])
	}};
}}
""".format(iconic_trads=",".join(map(lambda trad: "'" + trad + "'", iconic_trads)))
        )

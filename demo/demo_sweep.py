"""Demo sweep for learning rate and weight decay."""

import itertools

def make_cfgs() -> list[dict]:
	lrs = [1e-4, 3e-4, 1e-3, 3e-3]
	k = [16, 32, 64, 128]
	return [
		{"lr": lr, "sae": { "activation": {"top_k": k}, }}
		for lr, k in itertools.product(lrs, k)
	]

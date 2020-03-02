import flameflower.autograd.tensor_library as tl

def random_search(rng, shape, real_valued=True):
	if real_valued:
		generated = tl.random.uniform(*rng, size=shape)
	else:
		generated = tl.random.choice(tl.array(rng), size=shape)
	return generated



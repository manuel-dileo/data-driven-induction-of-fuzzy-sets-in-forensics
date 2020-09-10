def alpha_cut(alpha,target,item,est_mu,classes):
	return target if est_mu[target](item)>=alpha else 1-target
    

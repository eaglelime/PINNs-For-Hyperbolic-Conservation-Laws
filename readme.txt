


Ex param randomizer:

def random_int(min_val, max_val):
    return np.random.randint(min_val, max_val+1)

def random_log_float(min_val, max_val):
    return 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))

def random_float(min_val, max_val):
    return np.random.rand(1)[0]*(max_val-min_val)+min_val

def param_randomizer():
    ic_start_loss = random_float(4,6)
    ic_end_loss = ic_start_loss

    bc_start_loss = random_float(0.5,2)
    bc_end_loss = bc_start_loss

    entropy_start_loss = random_float(800,1200)
    entropy_end_loss = entropy_start_loss

    adv_entropy_start_loss = random_float(50,150)
    adv_entropy_end_loss = adv_entropy_start_loss

    start_lr = random_log_float(2e-3, 5e-2)
    end_lr = random_log_float(1e-4, start_lr)
    adv_start_lr = random_log_float(2e-3, 1e-2)
    adv_end_lr = random_log_float(1e-4, adv_start_lr)
    max_clip = random_float(0.5, 2)
    adv_max_clip = random_log_float(1e-1, 20)
    batch_size_entropy = random_int(8000,12000)
    width = random_int(10,30)
    depth = random_int(4,8)
    adv_width = random_int(5,15)
    adv_depth = random_int(2,4)

    adv_training_per_epoch = random_int(4,10)
    adv_reset_freq = random_int(100,400)

    return Hyperparams(
    ic_start_loss = ic_start_loss,
    ic_end_loss = ic_end_loss,
    bc_start_loss = bc_start_loss,
    bc_end_loss = bc_end_loss,
    entropy_start_loss = entropy_start_loss,
    entropy_end_loss = entropy_end_loss,
    adv_entropy_start_loss = adv_entropy_start_loss,
    adv_entropy_end_loss = adv_entropy_end_loss,
    start_lr = start_lr,
    end_lr = end_lr,
    adv_start_lr = adv_start_lr,
    adv_end_lr = adv_end_lr,
    max_clip = max_clip,
    adv_max_clip = adv_max_clip,
    batch_size_entropy = batch_size_entropy,
    width = width,
    depth = depth,
    adv_width = adv_width,
    adv_depth = adv_depth,
    adv_training_per_epoch = adv_training_per_epoch,
    adv_reset_freq = adv_reset_freq
    )

def param_randomizer2():
    return Hyperparams(width = random_int(10, 25),
                        depth = random_int(3,8),
                        adv_width = random_int(4, 12),
                        adv_depth = random_int(2,6))

def cauchy_density_loss_randomizer():
    return Hyperparams(cauchy_density_reg=random_log_float(0.001,10),
                        gamma = random_log_float(1e-2,5e-1))

if __name__ == "__main__":
    #hp = Hyperparams.from_json("/home/o/Documents/PINNs-for-hyperbolic-conservation-laws/output/weak_burger_continuous/20260506-081224/hyperparams.json")
    for i in range(1):
        #hp = cauchy_density_loss_randomizer()
        hp = Hyperparams()
        run_test(hp)













Another example:

def param_randomizer():
        ic_start_loss = utils.random_float(4,6)
        ic_end_loss = ic_start_loss

        bc_start_loss = utils.random_float(0.5,2)
        bc_end_loss = bc_start_loss

        entropy_start_loss = utils.random_float(800,1200)
        entropy_end_loss = entropy_start_loss

        adv_entropy_start_loss = utils.random_float(50,150)
        adv_entropy_end_loss = adv_entropy_start_loss

        start_lr = utils.random_log_float(2e-3, 5e-2)
        end_lr = utils.random_log_float(1e-4, start_lr)
        adv_start_lr = utils.random_log_float(2e-3, 1e-2)
        adv_end_lr = utils.random_log_float(1e-4, adv_start_lr)
        max_clip = utils.random_float(0.5, 2)
        adv_max_clip = utils.random_log_float(1e-1, 20)
        batch_size_entropy = utils.random_int(8000,12000)

        width = utils.random_int(10,30)
        depth = utils.random_int(4,8)
        adv_width = utils.random_int(5,15)
        adv_depth = utils.random_int(2,4)

        adv_training_per_epoch = utils.random_int(4,10)
        adv_reset_freq = utils.random_int(100,400)

        return Hyperparams(
        ic_start_loss = ic_start_loss,
        ic_end_loss = ic_end_loss,
        bc_start_loss = bc_start_loss,
        bc_end_loss = bc_end_loss,
        entropy_start_loss = entropy_start_loss,
        entropy_end_loss = entropy_end_loss,
        adv_entropy_start_loss = adv_entropy_start_loss,
        adv_entropy_end_loss = adv_entropy_end_loss,
        start_lr = start_lr,
        end_lr = end_lr,
        adv_start_lr = adv_start_lr,
        adv_end_lr = adv_end_lr,
        max_clip = max_clip,
        adv_max_clip = adv_max_clip,
        batch_size_entropy = batch_size_entropy,
        width = width,
        depth = depth,
        adv_width = adv_width,
        adv_depth = adv_depth,
        adv_training_per_epoch = adv_training_per_epoch,
        adv_reset_freq = adv_reset_freq
        )

def param_randomizer2():
        ent_start_loss = utils.random_log_float(10,1000)
        adv_entropy_start_loss = utils.random_log_float(1,1000)
        weak_integral_start_loss = utils.random_log_float(0.1,100)
        adv_weak_integral_start_loss = utils.random_log_float(1,1000)
        return Hyperparams(entropy_start_loss= ent_start_loss,
                entropy_end_loss = ent_start_loss,
                adv_entropy_start_loss = adv_entropy_start_loss,
                adv_entropy_end_loss = adv_entropy_start_loss,
                weak_integral_start_loss= weak_integral_start_loss,
                weak_integral_end_loss = weak_integral_start_loss,
                adv_weak_integral_start_loss= adv_weak_integral_start_loss,
                adv_weak_integral_end_loss = adv_weak_integral_start_loss)

def pert_randomizer():
    hp = Hyperparams()
    hp.start_lr = utils.perturb_log(hp.start_lr)
    hp.end_lr = utils.perturb_log(hp.end_lr)
    hp.ic_start_loss = utils.perturb_linear(hp.ic_start_loss)
    hp.ic_end_loss = hp.ic_start_loss
    hp.entropy_start_loss = utils.perturb_linear(hp.entropy_start_loss)
    hp.entropy_end_loss = hp.entropy_start_loss
    hp.adv_entropy_start_loss = utils.perturb_linear(hp.ic_start_loss)
    hp.adv_entropy_end_loss = hp.adv_start_lr
    hp.weak_integral_start_loss = utils.perturb_linear(hp.weak_integral_start_loss)
    hp.weak_integral_end_loss = hp.weak_integral_start_loss
    hp.adv_weak_integral_start_loss= utils.perturb_linear(hp.adv_weak_integral_start_loss)
    hp.adv_weak_integral_end_loss = hp.adv_weak_integral_start_loss
    hp.epochs = 2000
    return hp

def cauchy_density_loss_randomizer():
        return Hyperparams(cauchy_density_reg=utils.random_log_float(0.001,10),
                           gamma = utils.random_log_float(1e-2,5e-1))

if __name__ == "__main__":
        #hp = Hyperparams.from_json("/home/o/Documents/PINNs-for-hyperbolic-conservation-laws/output/weak_burger_continuous/20260506-081224/hyperparams.json")
        for i in range(1):
                #hp = cauchy_density_loss_randomizer()
                hp = pert_randomizer()
                run_test()
from tensorlayer.files import load_and_assign_npz_dict, save_npz_dict

def set_mode(nets, mode):
    if mode == "train":
        for net in nets:
            net.train()
    else:
        for net in nets:
            net.eval()
    return nets

def save(nets, name):
    save_npz_dict(nets[0].all_weights, name=name+"g_A_ae")
    save_npz_dict(nets[1].all_weights, name=name+"g_B_ae")
    save_npz_dict(nets[2].all_weights, name=name+"d_A")
    save_npz_dict(nets[3].all_weights, name=name+"d_B")
    save_npz_dict(nets[4].all_weights, name=name+"g_A")
    save_npz_dict(nets[5].all_weights, name=name+"g_B")
    

def load(checkpoint_name, nets):
    load_and_assign_npz_dict(checkpoint_name + "g_A_ae.npz", network=nets[0])
    load_and_assign_npz_dict(checkpoint_name + "g_B_ae.npz", network=nets[1])
    load_and_assign_npz_dict(checkpoint_name + "d_A.npz", network=nets[2])
    load_and_assign_npz_dict(checkpoint_name + "d_B.npz", network=nets[3])
    load_and_assign_npz_dict(checkpoint_name + "g_A.npz", network=nets[4])
    load_and_assign_npz_dict(checkpoint_name + "g_B.npz", network=nets[5])
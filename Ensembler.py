import numpy as np

npys = [np.array([
    "./data/features/testB_image_b4ns_sorted_all.npy",
    "./data/features/testB_image_ev2m_sorted_512_all.npy",
    "./data/features/testB_image_nfl1_sorted_all.npy",
    "./data/features/testB_image_200d_sorted_all.npy",
    "./data/features/testB_image_swb_sorted_all.npy",
]),np.array([
    "./data/features/testB_text_rbt_sorted_all.npy",
    "./data/features/testB_text_bt_sorted_all.npy",
    "./data/features/testB_text_btwwm_sorted_all.npy",
])]
preds = \
    np.stack([np.load(_) for _ in npys[0]]).mean(0) * 0.6 + \
    np.stack([np.load(_) for _ in npys[1]]).mean(0) * 0.4

np.save("./data/pseudo/90655.npy", preds)
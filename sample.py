#








def make_oneshot_task(self, N, s = "val"):

     indices    = rng.randint(0, n_examples, size = N)
     categories = rng.choice(range(n_classes), size = N, replace = False)
     (ex1, ex2) = rng.choice(n_examples, replace = False, size = 2)

     true_category = categories[0]

     test_image = np.asarray([ X[true_category, ex1, :, :] ] * N)  # クラスを選んで、そのex1番目のN個のコピーを作る

     support_set = X[categories, indices, :, :]

     support_set[0, :, :] = X[true_category, ex2]  # クラスを選んで、そのex２番目をindex 0にコピーする
                                                   # ex1のデータとは一致しなければならない

     targets = np.zeros(N)
     targets[0] = 1

     shuffle　　　　　　　　　　　　　　　　　　　　　　　 # 上のex1, ex2の順序をindex 0からシャッフルする

     return [test_image, support_set], targets



















#

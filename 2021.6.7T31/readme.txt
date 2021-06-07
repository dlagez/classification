# 直接使用t31数据
nuSVC   0.84375

random-ubder-sampling 

NuSVC(nu=0.3): 0.734375
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('linearsvc',
                 LinearSVC(max_iter=10000, random_state=0, tol=1e-05))]): 0.5
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))]): 0.515625

--------------------------------------------------------------------------------------
nearMiss

NuSVC(nu=0.3): 0.515625
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('linearsvc',
                 LinearSVC(max_iter=10000, random_state=0, tol=1e-05))]): 0.46875
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))]): 0.5

--------------------------------------------------------------------------------------

NuSVC(nu=0.3): 0.5
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('linearsvc',
                 LinearSVC(max_iter=10000, random_state=0, tol=1e-05))]): 0.53125
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))]): 0.5
--------------------------------------------------------------------------------------
BalancedBaggingClassifier  0.671875
--------------------------------------------------------------------------------------
EasyEnsembleClassifier   0.734375




nuSVC参数：

nu：float, default=0.5
边距误差分数的上限（参见用户指南）和支撑载体分数的下限。 应该是间隔（0,1]。

kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
指定要在算法中使用的内核类型。 它必须是“线性”，'poly'，'rbf'，'sigmoid'，'预染色'或可调用的一个。 
如果没有给出，将使用“RBF”。 如果给出了可调用的可调用，则用于预编译内核矩阵。

degreeint, default=3
多项式核函数 ('poly') 的次数。被所有其他内核忽略。

参考链接：https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC
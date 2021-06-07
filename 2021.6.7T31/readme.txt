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
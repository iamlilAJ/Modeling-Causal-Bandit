import  numpy as np

class LinUCB:
    def __init__(self, num_arms, num_features, alpha):
        self.num_arms = num_arms
        self.num_features = num_features
        self.alpha = alpha
        self.A = [np.eye(num_features) for _ in range(num_arms)]
        self.b = [np.zeros((num_features, 1)) for _ in range(num_arms)]
        self.theta = [np.zeros((num_features, 1)) for _ in range(num_arms)]


    def select_arm(self, context):
        context_update = np.array(context[:-1])

        p = [self.theta[a].T @ context_update + self.alpha * np.sqrt(context_update.T @ np.linalg.inv(self.A[a]) @ context_update)
             for a in range(self.num_arms)]

        return np.argmax(p)

    def update(self, arm, context, reward):
        context_update = np.array(context[:-1])
        context_update = context_update.reshape(-1, 1)

        self.A[arm] += context_update @ context_update.T

        self.b[arm] += reward * context_update
        self.theta[arm] = np.linalg.inv(self.A[arm]) @ self.b[arm]

class LinUCB_MB:
    def __init__(self, num_arms, num_features, alpha, adj):
        self.num_arms = num_arms
        self.num_features = num_features
        self.alpha = alpha
        self.A = [np.eye(num_features) for _ in range(num_arms)]
        self.b = [np.zeros((num_features, 1)) for _ in range(num_arms)]
        self.theta = [np.zeros((num_features, 1)) for _ in range(num_arms)]
        self.adj = adj

    def select_arm(self, context):
        context_update = context[:-1]
        p = [self.theta[a].T @ context_update + self.alpha * np.sqrt(context_update.T @ np.linalg.inv(self.A[a]) @ context_update)
             for a in range(self.num_arms)]
        return np.argmax(p)

    def update(self, arm, context, reward):

        context_update = context[:-1]
        context_update = context_update.reshape(-1, 1)

        self.A[arm] += context_update @ context_update.T

        self.b[arm] += reward * context_update
        self.theta[arm] = np.linalg.inv(self.A[arm]) @ self.b[arm]

    def mb_update(self, arm,context, reward, env):
        context_copy = context.copy()
        self.update(arm, context_copy, reward)

        intervened_context, _ = env.intervention(context, arm, 1)

        for i in range(arm+1, self.num_arms-1):
            if context_copy[i] == 0 and intervened_context[i] == 1:
                pesudo_context = intervened_context.copy()
                pesudo_context[i:] = [0] * (len(pesudo_context) - i)
                # pesudo_context[i] = 0
                # print(i)
                # print(intervened_context, pesudo_context)

                self.update(i, pesudo_context, reward)


from operator import itemgetter
import numpy as np
import math
import random

class pointing_agent():
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.1
        self.softmax_temp = 1.0
        self.q = {}
        self.learning = True
        self.log_p = False

        self.k_alpha = 0.12

        # Transitions for dyna. The transition table holds
        # <belief-action> -> <belief',reward,count> transitions.
        self.tr = {}

        # Transition tracker keeps record of when a state-action pair
        # has been visited the last time. This update needs an
        # external model time supplied.
        self.tr_tracker = {}
        self.dyna = 0 # how many dyna-iterations

        self.log_header = "trial instruction mt actions hit"
        self.log = []

        self.trial_n = 0

        self.actions = []
        # 0 = move, 1 = peck
        for action in [0,1]:
            # true sat
            for sat in [0.05, 1]:
                for direction in [0]:
                    for amplitude in [-20,-10,-5,-4,-3,-2,-1,0,1,2,3,4,5,10,20]:
                        # Must use repr to use in the belief table.
                        self.actions.append(repr([action,sat,direction,amplitude]))

    def clear(self):
        self.belief = None
        self.previous_action = None
        self.action = None
        # Desired sat is the weigh we give to mt vs. hit. Larger
        # values empasise accuracy, smaller speed.
        self.instruction = random.choice([0.1,0.25,0.5,0.75,0.9])
        # Target as tolerance (position must be between these when peck)
        self.target = [19,21]
        self.reward = 0

        self.mt = 0
        self.position = 0

        # For logging
        self.trial_n += 1
        self.total_actions = 0
        self.total_time = 0

    def point(self, debug = False):
        # TODO: two dimensions

        # Must listify and transform the string back to numers.
        self.action_l = list(map(float,self.action.replace('[','').replace(']','').replace(' ', '').split(',')))

        self.mt = WHo_mt(abs(self.action_l[3]),self.action_l[1])

        self.total_time += self.mt

        self.position += round(self.action_l[3] + np.random.normal(0,self.action_l[1]))

        if debug:
            print("Pointing action (peck = ", self.action_l[0], "), mt = ", self.mt, ", new pos = ", self.position,
                  sep = '')
            if self.action_l[0] == 1:
                if self.position > self.target[0] and self.position < self.target[1]:
                    print("Peck inside target zone.")


    def set_belief(self):
        self.previous_belief = self.belief
        self.belief = repr([self.instruction, self.position])
        if self.belief not in self.q:
            self.q[self.belief] = {}
            for a in self.actions:
                self.q[self.belief][a] = 0.0

    def calculate_reward(self, debug = False):
        self.reward = -self.mt*(1-self.instruction)
        # Is peck?
        peck_reward = 0
        if self.action_l[0] == 1:
            if self.position > self.target[0] and self.position < self.target[1]:
                self.reward += 1 * self.instruction
                peck_reward = 1
        if debug:
            print("Obtained reward ", self.reward, " (mt = ", -self.mt, " peck = ", peck_reward, "),",
                  sep = '')

    def update_q_sarsa(self, debug = False):
        if self.learning and self.previous_action:
            previous_q = self.q[self.previous_belief][self.previous_action]
            next_q = self.q[self.belief][self.action]
            self.q[self.previous_belief][self.previous_action] = \
                previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)
            if debug:
                print("Update:", self.previous_belief, self.previous_action, round(previous_q, 2), "->",
                      round(self.q[self.previous_belief][self.previous_action],2))

    def update_q_exp_sarsa(self, debug = False):
        if self.learning and self.previous_action:
            previous_q = self.q[self.previous_belief][self.previous_action]
            next_q = 0
            p = {}
            for a in self.q[self.belief].keys():
                p[a] = math.exp(self.q[self.belief][a] / self.softmax_temp)
            for a in self.actions:
                next_q += self.q[self.belief][a] * p[a]

            self.q[self.previous_belief][self.previous_action] = \
                previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)

    def update_q_td(self, debug = False):
        if self.learning and self.action:
            previous_q = self.q[self.belief][self.action]
            self.q[self.belief][self.action] = \
                previous_q + self.alpha * (self.reward - previous_q)
            if debug:
                print("Update q of", self.belief, self.action, round(previous_q, 2), "->",
                      round(self.q[self.belief][self.action],2))

    def calculate_max_q_value(self):
        return max(self.q[self.belief].items(), key=itemgetter(1))

    def update_q_learning(self, debug = False):
        if self.previous_action != None and self.learning:
            previous_q = self.q[self.previous_belief][self.previous_action]
            next_q = self.calculate_max_q_value()[1]
            self.q[self.previous_belief][self.previous_action] = \
                previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)        

    def choose_action_epsilon(self, debug = False):
        self.previous_action = self.action
        rand = False
        if random.random() < self.epsilon:
            self.action = np.random.choice(self.actions)
            rand = True
        else:            
            self.action = max(self.q[self.belief].items(), key=itemgetter(1))[0]
        if debug:
            print("In state ", self.belief, " selecting action ", self.action, " (random action = ", rand, ")",
                  sep = '')

    def weighted_random(self, weights):
        number = random.random() * sum(weights.values())
        for k,v in weights.items():
            if number < v:
                break
            number -= v
        return k

    def choose_action_softmax(self, debug = False):
        self.previous_action = self.action
        if self.softmax_temp == 0:
            self.action = max(self.q[self.belief].items(), key=itemgetter(1))[0]
            if debug:
                print("In state ", self.belief, " selecting action ", self.action, " (random action = False)",
                      sep = '')
            return
        p = {}
        for a in self.q[self.belief].keys():
            p[a] = math.exp(self.q[self.belief][a] / self.softmax_temp)
        s = sum(p.values())
        if s != 0:
            p = {k: v/s for k,v in p.items()}
            self.action = self.weighted_random(p)
        else:
            self.action = np.random.choice(list(p.keys()))
        if debug:
            print("In state ", self.belief, " selecting action ", self.action,
                  sep = '')

    def update_dyna(self):
        # Update model.
        if self.learning and self.previous_action:
            modeltime = None
            # If modeltime is supplied, add it to tracker and later
            # increase reward based on how long since this sa-pair has
            # been updated.
            if modeltime:
                if self.previous_belief not in self.tr_tracker:
                    self.tr_tracker[self.previous_belief] = {}
                self.tr_tracker[self.previous_belief][self.previous_action] = modeltime


            if self.previous_belief not in self.tr:
                self.tr[self.previous_belief] = {}
            if self.previous_action not in self.tr[self.previous_belief]:
                self.tr[self.previous_belief][self.previous_action] = {}
            if self.belief not in self.tr[self.previous_belief][self.previous_action]:
                self.tr[self.previous_belief][self.previous_action][self.belief] = [self.reward]
            else:
                self.tr[self.previous_belief][self.previous_action][self.belief].append(self.reward)


            for i in range(0, self.dyna):
                b1 = random.choice(list(self.tr.keys()))
                a = random.choice(list(self.tr[b1].keys()))
                b2 = random.choice(list(self.tr[b1][a].keys()))
                r  = np.mean(self.tr[b1][a][b2])
                if modeltime:
                    r += 0.01 * math.sqrt(modeltime - self.tr_tracker[b1][a])
                next_q = self.q[b1][a]
                max_q = max(self.q[b2].items(), key=itemgetter(1))[1]
                # Update q based on simulated experience.
                self.q[b1][a] = self.q[b1][a] + self.alpha * (r + self.gamma * max_q - self.q[b1][a])


    def do_iteration(self, debug = False):

        self.set_belief()
        self.choose_action_epsilon(debug = debug)
        self.update_q_learning(debug = debug)
        self.update_dyna()

        self.total_actions += 1        
        self.point(debug = debug)
        self.calculate_reward(debug = debug)
        
        # Task finished?
        if self.action_l[0] == 1:
            self.update_q_td(debug = debug)
            if self.log_p:
                hit = 0
                if self.position > self.target[0] and self.position < self.target[1]:
                    hit = 1
                self.log.append("{} {} {} {} {}".
                                format(self.trial_n, self.instruction, self.total_time, self.total_actions, hit))
            self.task_finished = True
            self.clear()        


    def do_task(self, debug = False):
        self.do_iteration(debug = debug)
        while self.belief:
            self.do_iteration(debug = debug)

    def write_log_to_file(self, filename):
        out = open(filename, "w")
        out.write(self.log_header + "\n")
        for d in self.log:
            out.write(d + "\n")
        out.close()

def WHo_mt(distance, sigma, k_alpha = 0.12):
    x0 = 0.092
    y0 = 1
    alpha = 0.5
    x_min = 0.006
    x_max = 0.06

    if distance == 0:
        distance = 0.0000001

    mt = pow((k_alpha * pow(((sigma - y0) / distance),(alpha - 1))), 1 / alpha ) + x0

    return mt

def WHo_pointing(distance, key_width, sigma, k_alpha = 0.12):
    #mt = np.ones((10000,)) * sigma
 
    mt = WHo_mt(distance, sigma, k_alpha)
 
    endpoint = np.abs(np.random.normal(0, sigma, 10000))
 
    hit = (endpoint <= (key_width / 2)).mean()

    return mt, hit, np.mean(endpoint)
    #return np.mean(mt), hit, np.mean(endpoint), np.max(endpoint)
    # mt = []
    # hit = []
    # for i in range(0,10000):
    #     mt.append(WHo_mt(distance, sigma))
    #     if np.random.normal(0, sigma*distance) > key_width / 2:
    #         hit.append(0)
    #     else:
    #         hit.append(1)
    # return np.mean(mt), np.mean(hit)


# Return hit percentage, target size and enpoint variation (sd)
def hit_from_sd(target_size, sd):
    return math.erf((target_size/2/sd)/math.sqrt(2))

# a = pointing_agent()
# a.clear()
#a.dyna = 1000
# episodes = 5000000
# start_temp = 1
# a.log_p = True
# a.softmax_temp = start_temp
# for e in tqdm.tqdm(range(episodes), position = 0, leave = True):
#     a.do_task()    
# a.clear()
# a.learning = False
# a.softmax_temp = 0
# a.epsilon = 0
# a.log_p = True
# for e in range(10000):
#     a.do_task()
# a.write_log_to_file("out.csv")


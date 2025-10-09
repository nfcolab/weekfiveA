import random
import ui
import numpy as np
import math
from operator import itemgetter

class decision_task():

    def __init__(self, ui, colours = True, time_cost = 1):

        self.ui = ui

        self.colours = colours
        self.time_cost = time_cost

        self.encoding_penalty = 0

        self.actions = []
        for e in ui.elements:
            self.actions.append(e)

        self.actions.append("accept")
        self.actions.append("reject")

        self.q = {}

        self.starting_loc = ui.closest_element([0,0])

        self.learning = True

        self.state = {}

        self.correct_reward = 20
        self.incorrect_reward = 0

        self.alpha = 0.1 # learning rate
        self.epsilon = 0.1 # explore vs exploit
        self.gamma = 0.9 # future reward discount
        self.softmax_temp = 1

        self.clear()

    def clear(self):

        self.randomise_decision_ui(colours = self.colours)

        self.visual_matrix = {}
        for e in self.ui.elements:
            self.visual_matrix[e] = [0,None]

        self.eye_loc = random.choice(list(self.ui.elements.keys()))
        self.scanpath = [self.eye_loc]

        self.mt = 0
        self.task_time = 0
        self.terminal = False

        self.previous_state = None
        self.previous_action = None
        self.action = None
        self.set_state()

    def set_state(self):
        self.current_state = repr(self.visual_matrix)
        if self.current_state not in self.q:
            self.q[self.current_state] = {}
            # Add all actions as possible pairs if this new state.
            for a in self.actions:
                self.q[self.current_state][a] = 0.0

    def update_q_learning(self):
        # Only learn if there is a previous action. If this is a start
        # of a new episode after a self.clear(), cannot learn yet.
        if self.previous_action != None:
            previous_q = self.q[self.previous_state][self.previous_action]
            next_q = max(self.q[self.current_state].items(), key = itemgetter(1))[1]
            self.q[self.previous_state][self.previous_action] = \
                previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)

    def update_q_td(self):
        previous_q = self.q[self.current_state][self.action]
        self.q[self.current_state][self.action] = \
                previous_q + self.alpha * (self.reward - previous_q)


    def choose_action_epsilon_greedy(self):
        if random.random() < self.epsilon:
            self.action = random.choice(self.actions)
            return "randomly" # for output (debug) purposes
        else:
            self.action = max(self.q[self.current_state].items(), key = itemgetter(1))[0]
            return "greedily"

    def choose_action_softmax(self, debug = False):
        p = {}
        for a in self.q[self.current_state].keys():
            p[a] = math.exp(self.q[self.current_state][a] / self.softmax_temp)
        s = sum(p.values())
        if debug:
            print(p)
        if s != 0:
            p = {k: v / s for k, v in p.items()}
            if debug:
                print(p)
            self.action = self.weighted_random(p)
        else:
            self.action = np.random.choice(list(p.keys()))

    def weighted_random(self, weights):
        number = random.random() * sum(weights.values())
        for k, v in weights.items():
            if number < v:
                break
        return k


    def calculate_reward(self):
        #self.reward = -self.mt
        self.reward = 0
        if self.action == "accept":
            if self.correct_answer == 1:
                self.reward += self.correct_reward - self.task_time*self.time_cost
            else:
                self.reward += self.incorrect_reward  - self.task_time*self.time_cost
        if self.action == "reject":
            if self.correct_answer == 0:
                self.reward += self.correct_reward - self.task_time*self.time_cost
            else:
                self.reward += self.incorrect_reward - self.task_time*self.time_cost

    def do_step(self, print_progress = False, force_action = None):
        self.previous_state = self.current_state
        self.previous_action = self.action
        self.set_state()

        if print_progress:
            print("Now in state:", self.current_state)

        if self.learning:
            self.update_q_learning()

        if not force_action:
            how = self.choose_action_epsilon_greedy()
            # how = "softmax"
            # self.choose_action_softmax()
        else:
            how = "forced"
            self.action = force_action

        if print_progress: print("Took action <", self.action, "> ", how, sep = '')

        if self.action in self.ui.elements:
            self.mt, moved = self.ui.emma_time(self.action, eye_loc = self.eye_loc)
            self.mt += self.encoding_penalty
            if moved:
                self.eye_loc = self.action
            self.scanpath.append(self.action)
            if self.ui.elements[self.action].data >= 0.5:
                self.visual_matrix[self.action] = [1,"green"]
            else:
                self.visual_matrix[self.action] = [1,"red"]
            if self.colours:
                for e in self.ui.elements:
                    if self.ui.element_distance(e,self.action)/self.ui.element_size(e) < 2.5:
                        self.visual_matrix[e] = [self.visual_matrix[e][0],self.ui.elements[e].color]
        else:
            self.mt = 0.5
            self.terminal = True

        self.task_time += self.mt

        self.calculate_reward()

        if self.terminal:
            if print_progress:
                print("Task done!")
            if self.learning:
                self.update_q_td()

    def randomise_decision_ui(self, colours = True):
        n_elements = len(self.ui.elements)
        correct_answer = random.randint(0,1)

        for e in self.ui.elements:
            self.ui.elements[e].data = 0

        if correct_answer == 1:
            greens = random.randint(round(n_elements/2)+1,n_elements-1)
        else:
            greens = random.randint(0,round(n_elements/2)-1)

        for e in random.sample(list(self.ui.elements.keys()), greens):
            if colours: self.ui.elements[e].color = "green"
            else: self.ui.elements[e].color = "grey"
            self.ui.elements[e].data = random.randint(51,99)/100

        for e in self.ui.elements:
            if self.ui.elements[e].data < 0.5:
                if colours: self.ui.elements[e].color = "red"
                else: self.ui.elements[e].color = "grey"
                self.ui.elements[e].data = random.randint(1,49)/100

        self.correct_answer = correct_answer

m = 8
decision_ui = ui.ui()
decision_ui.add_element("e1", 0*m,0*m,50*m,50*m)
decision_ui.add_element("e2", 75*m,0*m,50*m,50*m)
decision_ui.add_element("e3", 150*m,0*m,50*m,50*m)

decision_ui.add_element("e4", 0*m,75*m,50*m,50*m)
decision_ui.add_element("e5", 75*m,75*m,50*m,50*m)
decision_ui.add_element("e6", 150*m,75*m,50*m,50*m)

decision_ui.add_element("e7", 0*m,150*m,50*m,50*m)
decision_ui.add_element("e8", 75*m,150*m,50*m,50*m)
decision_ui.add_element("e9", 150*m,150*m,50*m,50*m)

def train_decision_maker(ui, colours = True, time_cost = 1, incorrect_reward = 0, encoding_penalty = 0, episodes = 800000):
    print("Starting training...")
    agent = decision_task(ui, colours, time_cost)
    agent.encoding_penalty = encoding_penalty
    agent.incorrect_reward = incorrect_reward
    i = 0
    until = episodes
    rewards = []
    while i < until:
        agent.do_step()
        if agent.terminal:
            rewards.append(agent.reward)
            agent.clear()
            i+=1
            if i%round(until/10)==0:
                print(round(i/until,1),round(np.mean(rewards),1))
                rewards = []
    print("Training finished.")
    return agent

def simulate_decision_task(agent, element_values = None):
    agent.epsilon = 0
    agent.learning = False
    agent.clear()
    if element_values:
        reds = 0
        for e in element_values:
            agent.ui.elements[e].data = element_values[e]
            if agent.ui.elements[e].data < 0.5:
                reds += 1
            if agent.colours:
                if agent.ui.elements[e].data < 0.5:
                    agent.ui.elements[e].color = "red"
                else:
                    agent.ui.elements[e].color = "green"
            else:
                agent.ui.elements[e].color = "grey"
        if reds/len(agent.ui.elements)>0.5:
            agent.correct_answer = 0
        else:
            agent.correct_answer = 1

    for i in range(100):
        agent.do_step()
        if agent.terminal:
            break

    if not agent.terminal:
        print("Warning: the model got stuck! Please rerun, and if you get this warning often, consider retraining the model.")
        return None

    correctly = "correctly"
    if (agent.correct_answer == 0 and agent.action == "accept") or (agent.correct_answer == 1 and agent.action == "reject"):
        correctly = "incorrectly"
    print("Decision was", correctly, agent.action, "and it took", round(agent.task_time,1), "seconds.")
    ui.visualise_UI(agent.ui, scanpath = agent.scanpath)

#simulate_decision_task(agent)

def sample_decision_agent(agent, n = 1000):
    correct_answers = []
    rewards = []
    task_times = []
    fixations = []

    agent.epsilon = 0
    agent.learning = False
    agent.clear()


    for i in range(n):
        for i in range(100):
            agent.do_step()
            if agent.terminal:
                break
        if agent.terminal:
            if (agent.correct_answer == 0 and agent.action == "accept") or (agent.correct_answer == 1 and agent.action == "reject"):
                correct_answers.append(0)
            else:
                correct_answers.append(1)
                rewards.append(agent.reward)
                fixations.append(len(agent.scanpath))
                task_times.append(agent.task_time)
        agent.clear()

    return np.mean(correct_answers), round(np.mean(task_times),1), round(np.mean(fixations),1)


# agent = train_decision_maker(decision_ui, colours = False)
# agent_col = train_decision_maker(decision_ui, colours = True)

# agent_rev = train_decision_maker(decision_ui, colours = False, incorrect_reward = -10)
# agent_col_rev = train_decision_maker(decision_ui, incorrect_reward = -10)

# agent_vis = train_decision_maker(decision_ui, colours = False, time_cost = 5)
# agent_col_vis = train_decision_maker(decision_ui, colours = True, time_cost = 5)

# agent_time = train_decision_maker(decision_ui, colours = False, encoding_penalty = 1)
# agent_col_time = train_decision_maker(decision_ui, colours = True, encoding_penalty = 1)

# print(sample_decision_agent(agent))
# print(sample_decision_agent(agent_col))
# print(sample_decision_agent(agent_rev))
# print(sample_decision_agent(agent_col_rev))
# print(sample_decision_agent(agent_vis))
# print(sample_decision_agent(agent_col_vis))

task1 = {
    "e1": 0.3,
    "e2": 0.6,
    "e3": 0.8,
    "e4": 0.7,
    "e5": 0.3,
    "e6": 0.6,
    "e7": 0.2,
    "e8": 0.5,
    "e9": 0.4
}

task2 = {
    "e1": 0.3,
    "e2": 0.2,
    "e3": 0.3,
    "e4": 0.3,
    "e5": 0.2,
    "e6": 0.4,
    "e7": 0.7,
    "e8": 0.3,
    "e9": 0.6
}

task3 = {
    "e1": 0.7,
    "e2": 0.8,
    "e3": 0.6,
    "e4": 0.7,
    "e5": 0.9,
    "e6": 0.4,
    "e7": 0.7,
    "e8": 0.3,
    "e9": 0.6
}

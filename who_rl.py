from operator import itemgetter # getting max q from a dict
import math
import random
import pointing
import ui
import numpy as np

class who_rl():
    def __init__(self, ui, who_alpha = 0.12, miss_penalty = 0, e3_reward = 1, e4_reward = 1):
        self.actions = []
        possible_actions = []
        possible_new_actions = ["e1","e2","e3","e4"]
        for a in possible_new_actions:
            if a in ui.elements:
                possible_actions.append(a)

        for a in possible_actions:
            for i in ["1","2","3","4","5"]:
                self.actions.append(a + "-" + i)

        self.ui = ui

        self.e3_reward = e3_reward
        self.e4_reward = e4_reward

        self.who_alpha = who_alpha
        self.miss_penalty = miss_penalty

        self.sigmas = {
            "1": 5,
            "2": 60,
            "3": 120,
            "4": 800,
            "5": 1000}

        # self.sigmas = {
        #     "1": 0.1,
        #     "2": 0.5,
        #     "3": 1,
        #     "4": 2,
        #     "5": 10}

        self.instructions = {
            "1": "MA",
            "2": "A",
            "3": "B",
            "4": "S",
            "5": "MS"}

        # Q table will be populated with state-action-value triplets.
        # It is a dictionary, with state as a key, and a new
        # dictionary as its value. This new dictionary contains
        # actions as keys, and state-action values as values.
        self.q = {}

        # The state is a dictionary that can be used to represent the
        # environment. The dictionary format allows a human-readable
        # representation of the different components of the full
        # state. See clear() to get a glimpse of the state space.
        self.state = {}

        # RL parameters.
        self.alpha = 0.1 # learning rate
        self.epsilon = 0.1 # explore vs exploit
        self.gamma = 0.9 # future reward discount

        # Initialise the environment.
        self.clear()

    # In order to learn, the agent needs to do the task multiple
    # times. This function clears the environment to its starting
    # stage. Note that the Q table is not re-initialised: this allows
    # the agent to carry information from previous task iterations.
    def clear(self):
        self.state = {
            "cursor_pos": "e1",
            "pressed_buttons": "None"
            }

        self.pressed_buttons = []

        self.instruction = ""

        self.mt = 0
        self.total_mt = 0
        self.offset = 0

        self.hit = False

        self.terminal = False

        # Start with no previous state, no current state, no previous
        # action, and no current action.
        self.previous_state = None
        self.current_state = None
        self.previous_action = None
        self.action = None

    # The logic of the environment. Given an action and a state,
    # change the state.
    def update_environment(self, print_progress = False):
        # Determine if the movement was a hit
        sigma = self.sigmas[self.action[3]]
        self.instruction = self.instructions[self.action[3]]
        target = self.action[0:2]
        target_size = self.ui.elements[target].max_size()

        if target == self.state["cursor_pos"]:
            #distance = max(self.offset, 1)
            distance = self.ui.elements[target].max_size()
            #distance = 1
        else:
            distance = self.ui.element_distance(self.state["cursor_pos"], target)

        self.mt = pointing.WHo_mt(distance, sigma, k_alpha = self.who_alpha)
        if print_progress:
            print("mt =", self.mt, "distance =", distance, "sigma =", sigma, "target_size =", target_size)

        # self.offset = np.random.normal(0, sigma/target_size)
        # if abs(self.offset) > 0.5:
        #     self.hit = False
        #     # self.target = self.ui.closest_element([self.ui.elements[target].x + self.offset*target_size,
        #     #                                        self.ui.elements[target].y + self.offset*target_size])
        # else:
        #     self.hit = True
        

        self.hit = False
        if random.random() < pointing.hit_from_sd(target_size, sigma):
            self.hit = True
            self.offset = 1
        if not self.hit:
            self.offset = sigma
        # if self.hit == False:
        #     offset_x = self.ui.elements[target].x_size
        #     if (random.random() - 0.5) < 0:
        #         dir_x = -random.random()*1.2 - offset_x
        #     else:
        #         dir_x = random.random()*1.2 + offset_x
        #     offset_y = self.ui.elements[target].y_size
        #     if (random.random() - 0.5) < 0:
        #         dir_y = -random.random()*1.2 - offset_y
        #     else:
        #         dir_y = random.random()*1.2 + offset_y

        #     loc = self.ui.elements[target].loc()

        #     cursor_pos = [loc[0] + round(dir_x), loc[1] + round(dir_y)]

        #     target = self.ui.closest_element(cursor_pos, elements_tabu = [target])


        if self.hit and self.action not in self.pressed_buttons:
            if target == "e3" and "e1" in self.pressed_buttons and "e2" in self.pressed_buttons:
                self.pressed_buttons.append(target)
            if target == "e2" and "e1" in self.pressed_buttons:
                self.pressed_buttons.append(target)
            if (target == "e4" and "e1" in self.pressed_buttons) or target == "e1":
                self.pressed_buttons.append(target)

        self.state["pressed_buttons"] = repr(self.pressed_buttons)
        self.state["cursor_pos"] = target

        self.total_mt += self.mt

        return self.mt, self.hit


    # Calculate the reward that the agent gets, given a state. Punish
    # for time consuming actions, reward for getting the task done.
    def calculate_reward(self):
        self.reward = 0
        self.reward = -self.mt
        if not self.hit:
            self.reward = self.miss_penalty + self.reward
        if "e3" in self.pressed_buttons:
            self.reward = 1*self.e3_reward + self.reward
            self.terminal = True
        if "e4" in self.pressed_buttons:
            self.reward = 1*self.e4_reward + self.reward
            self.terminal = True
        return self.reward

    # Epsilon greedy action selection.
    def choose_action_epsilon_greedy(self):
        if random.random() < self.epsilon:
            self.action = random.choice(self.actions)
            return "randomly" # for output (debug) purposes
        else:
            self.action = max(self.q[self.current_state].items(), key = itemgetter(1))[0]
            return "greedily"

    # For output (debug) purposes, cleanly print the state dictionary.
    def print_state(self):
        print("Current state:")
        for s in self.state:
            print("   ", s, ":", self.state[s])

    # For output (debug) purposes, cleanly print the nested Q
    # dictionary. Optionally, only print a given state's
    # action-values.
    def print_q(self, state = None):
        for s in self.q:
            if state == None or repr(state) == s:
                print(s)
                for a in self.actions:
                    print("   ", a, ":", round(self.q[s][a],2))

    # Update the q-table. Q learning can only be called after an
    # action has been taken, the environment transitioned, and the
    # reward observed. The easiest to do this is at the start of the
    # new iteration, because it lets us get the current max Q -value,
    # which can be then chained to the previous state-action Q-value,
    # which is the one that is to be updated.
    def update_q_learning(self):
        # Only learn if there is a previous action. If this is a start
        # of a new episode after a self.clear(), cannot learn yet.
        if self.previous_action != None:
            previous_q = self.q[self.previous_state][self.previous_action]
            next_q = max(self.q[self.current_state].items(), key = itemgetter(1))[1]
            self.q[self.previous_state][self.previous_action] = \
                previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)

    # Because Q-learning function updates Q-values only after seeing
    # the next state, it is not useful for training the last state of
    # an episode. For this purpose, before clearting the environment,
    # call td-update, which works like Q-learning but does not
    # consider the next state (which does not exist).
    def update_q_td(self):
        previous_q = self.q[self.current_state][self.action]
        self.q[self.current_state][self.action] = \
                previous_q + self.alpha * (self.reward - previous_q)

    # Do one iteration of the model.
    #
    # print_progress: print debug information
    #
    # force_action: instead of epsilon greedy action selection, force an action
    def do_step(self, print_progress = False, force_action = None):
        self.previous_state = self.current_state
        self.previous_action = self.action
        # Make the current state to string with repr(), so that the
        # current state can be accessed as a dictionary keyword. Note
        # that self.state and self.current_state are both the same
        # state, but in different data structures (dict vs string).
        self.current_state = repr(self.state)
        # Add the current state to the q table if it is not there yet.
        # Initialise all state-action pair values to 0.
        if self.current_state not in self.q:
            self.q[self.current_state] = {}
            # Add all actions as possible pairs if this new state.
            for a in self.actions:
                self.q[self.current_state][a] = 0.0

        if print_progress:
            print("Now in pos:", self.state["cursor_pos"])

        # Update the Q table based on previous state and previous
        # action, and the best (optimal) action from the current
        # state. If this is the first state (no previous state
        # exists), the function does not do anything.
        self.update_q_learning()

        # Choose action, store the explore or exploit as a string for
        # outputting (debug). If the action is forced, take that action.
        if force_action:
            self.action = force_action
            how = "forced"
        else:
            how = self.choose_action_epsilon_greedy()

        if print_progress: print("Took action <", self.action, "> ", how, sep = '')
        # Based on the action and the current state, update the environment.
        self.update_environment(print_progress)
        # Based on the new state after the update, observe the reward.
        # Note that learning only happens after the fact: either in
        # the beginning of next iteration, when the next state is
        # known (necessary for Q-learning), or if this is the last
        # state before clear, at the end of this iteration.
        self.calculate_reward()
        if print_progress:
            print("mt:",self.mt,"/ hit:",self.hit)
            print("Reward =", self.reward)

        # If the state is end state, update using TD learning, because
        # Q learning only happens when there is a previous state,
        # which clear removes.
        if self.terminal:
            if print_progress:
                print("Task done!")
                self.print_state()
            self.update_q_td()

# Create the ui
who_ui = ui.ui()

who_ui.add_element("e1", 10, 10, 10, 10, color = "grey")
who_ui.add_element("e2", 50, 50, 80, 80, color = "green")
who_ui.add_element("e3", 150, 200, 10, 10, color = "blue")
#who_ui.add_element("e4", 50, 300, 25, 20, color = "green")

# Create the agent.
def simulate_who_task(ui, who_alpha = 0.12, miss_penalty = 0, e3_reward = 1, e4_reward = 1, episodes = 100000):
    agent = who_rl(ui, who_alpha = who_alpha, miss_penalty = miss_penalty, e3_reward = e3_reward, e4_reward = e4_reward)
    # Learn the model multiple times. This takes some seconds to run.
    i = 0
    while i < episodes:
        agent.do_step()
        if agent.terminal:
            agent.clear()
            i += 1

    # Simulate converged behaviour once
    agent.epsilon = 0
    agent.clear()
    path = []
    mt = []
    for i in range(10):
        agent.do_step(print_progress = False)
        cursor_pos = agent.state["cursor_pos"]
        # If no hit, add noisy hit for visualisation purposes. Does
        # not affect the model.
        if agent.hit == False:
            offset_x = agent.ui.elements[cursor_pos].x_size
            if (random.random() - 0.5) < 0:
                dir_x = -random.random()*1.2 - offset_x
            else:
                dir_x = random.random()*1.2 + offset_x
            offset_y = agent.ui.elements[cursor_pos].y_size
            if (random.random() - 0.5) < 0:
                dir_y = -random.random()*1.2 - offset_y
            else:
                dir_y = random.random()*1.2 + offset_y

            loc = agent.ui.elements[cursor_pos].loc()

            cursor_pos = [loc[0] + round(dir_x), loc[1] + round(dir_y)]

        path.append(cursor_pos)
        mt.append([round(agent.total_mt,2), agent.instruction])
        if agent.terminal:
            break
    return path, mt

def visualise_who_task(who_ui, who_alpha = 0.12, miss_penalty = 0, e3_reward = 1, e4_reward = 1, episodes = 100000):
    path, mt = simulate_who_task(who_ui, who_alpha = who_alpha, miss_penalty = miss_penalty, e3_reward = e3_reward, e4_reward = e4_reward, episodes = episodes)
    print("Total number of pointing actions:", len(path))
    print("Pointing times:", mt)
    print("Total pointing time:", mt[-1][0])
    ui.visualise_UI(who_ui, path = path, annotate = mt)

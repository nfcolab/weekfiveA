# Create simple abstract UIs and explore task completion times using
# sequential Fitts' law.

import math

class element:
    def __init__(self, name, x, y, x_size, y_size, color = "grey", frequency = 0.1):
        self.name = name
        self.x = x
        self.y = y
        self.x_size = x_size
        self.y_size = y_size
        self.color = color

        self.data = None

        self.frequency = frequency

    # Return middle coordinate
    def loc(self):
        return [round(self.x+self.x_size/2),round(self.y+self.y_size/2)]

    def max_size(self):
        return max(self.x_size, self.y_size)    

class ui:
    def __init__(self, x_size = 100, y_size = 100):
        self.x_size = x_size
        self.y_size = y_size

        self.elements = {}
        self.distances = {}

        self.fitts_a = 0.230
        self.fitts_b = 0.166

        self.calibrate_user_distance()
        self.eye_loc = None

        self.ltm_pos = {}
        self.ltm_pos_fact = {}
        self.ltm_color = {}
        self.ltm_color_fact = {}


    def calibrate_UI_size(self):
        max_x = 0
        max_y = 0
        for e in self.elements:
            if self.elements[e].x + self.elements[e].x_size > max_x:
                max_x = self.elements[e].x + self.elements[e].x_size
            if self.elements[e].y + self.elements[e].y_size > max_y:
                max_y = self.elements[e].y + self.elements[e].y_size

        if max_x > self.x_size:
            #print("Note: adjusting UI x size from", self.x_size, "to", max_x)
            self.x_size = max_x
            self.calibrate_user_distance()

        if max_y > self.y_size:
            #print("Note: adjusting UI y size from", self.y_size, "to", max_y)
            self.y_size = max_y
            self.calibrate_user_distance()

    def learn_all_elements(self, expertise):
        for e in self.elements:
            self.learn_element_pos(e, expertise)
            self.learn_element_color(e, expertise)
            
    def learn_element_pos(self, element, expertise):
        if isinstance(expertise, str):
            if expertise == "intermediate":
                expertise = -0.75
            elif expertise == "expert":
                expertise = 1
        self.ltm_pos[element] = expertise
        self.ltm_pos_fact[element] = self.elements[element].loc()

    def learn_element_color(self, element, expertise):
        if isinstance(expertise, str):
            if expertise == "intermediate":
                expertise = -0.75
            elif expertise == "expert":
                expertise = 1
        self.ltm_color[element] = expertise
        self.ltm_color_fact[element] = self.elements[element].color

    def recall_time(self, activation):
        return 1.06*math.exp(-1.53*activation)

    # User is expected to "sit" in front of the device at a distance
    # that is 2.5 times the longer side of the device dimensions.
    def calibrate_user_distance(self):
        self.user_distance = max(self.x_size,self.y_size)*2

    def add_element(self, name, x, y, x_size, y_size, color = "grey", frequency = 0.1):
        e = element(name, x, y, x_size, y_size, color, frequency)
        self.elements[e.name] = e
        self.calibrate_UI_size()
        if not self.eye_loc:
            self.eye_loc = e.name

    def swap_elements(self, e1, e2):
        x_1 = self.elements[e1].x
        y_1 = self.elements[e1].y
        self.elements[e1].x = self.elements[e2].x
        self.elements[e1].y = self.elements[e2].y
        self.elements[e2].x = x_1
        self.elements[e2].y = y_1

    def modify_element(self, name, var, val):
        if var == 'x':
            self.elements[name].x = val
        if var == 'y':
            self.elements[name].y = val
        if var == 'x_size':
            self.elements[name].x_size = val
        if var == 'y_size':
            self.elements[name].y_size = val
        if var == 'color':
            self.elements[name].color = val

    # Given two elements or their names, return the distance between them
    def element_distance(self, element1, element2):
        # if isinstance(element1, element):
        #     element1 = element1.name
        # if isinstance(element2, element):
        #     element2 = element2.name
        # if (element1 in self.distances and element2 in self.distances[element1]) or \
        #    (element2 in self.distances and element1 in self.distances[element2]):
        #     return self.distances[element1][element2]

        loc1 = self.elements[element1].loc()
        loc2 = self.elements[element2].loc()
        dist = math.sqrt(math.pow(loc1[0] - loc2[0], 2) +
                         math.pow(loc1[1] - loc2[1], 2))
        # if element1 not in self.distances:
        #     self.distances[element1] = {}
        # self.distances[element1][element2] = dist
        return dist

    def element_size(self, element):
        return max(self.elements[element].x_size, self.elements[element].y_size)
    

    def closest_element(self, loc, elements = None):
        if not elements:
            elements = list(self.elements.keys())
        closest_e = None
        d = None
        for e in elements:
            loc2 = self.elements[e].loc()
            dd = math.sqrt(math.pow(loc[0] - loc2[0], 2) +
                           math.pow(loc[1] - loc2[1], 2))
            if not closest_e or dd < d:
                closest_e = e
                d = dd

        return closest_e

    # Calculate visual distance, as degrees, between two elements.
    def visual_distance(self, element1, element2):
        return 180 * (math.atan(self.element_distance(element1, element2) / self.user_distance) / math.pi)

    # Given size in pixels, what is the angular size.
    def angular_size(self, size):
        return 180 * (math.atan(size / self.user_distance) / math.pi)


    def fitts_movement_time(self, elements):
        mt = 0
        for e in range(len(elements)-1):
            mt += fitts_mt(self.element_distance(self.elements[elements[e]].name, self.elements[elements[e+1]].name), min(self.elements[elements[e+1]].x_size, self.elements[elements[e+1]].y_size), self.fitts_a, self.fitts_b)
        return mt

    def emma_time(self, target, eye_loc = None):
        if not eye_loc:
            eye_loc = self.eye_loc
        dist = self.visual_distance(eye_loc, target)
        return self.EMMA_fixation_time(dist, freq=self.elements[target].frequency)

    # Eye movement and encoding time come from EMMA (Salvucci, 2001). Also
    # return if a fixation occurred.
    def EMMA_fixation_time(self, distance, freq = 0.1):
        emma_KK = 0.006
        emma_k = 0.4
        emma_prep = 0.135
        emma_exec = 0.07
        emma_saccade = 0.002
        E = emma_KK * -math.log(freq) * math.exp(emma_k * distance)
        if E < emma_prep: return E, False
        S = emma_prep + emma_exec + emma_saccade * distance
        if (E <= S): return S, True
        E_new = (emma_k * -math.log(freq))
        T = (1 - (S / E)) * E_new
        return S + T, True

    # Given a list of element names, return a scanpath (as element
    # names) and total movement time.
    def create_emma_scanpath(self, elements):
        mt = 0
        fixations = [elements[0]]
        for e in range(len(elements)-1):
            mt_, moved = self.emma_time(elements[e+1], eye_loc = fixations[-1])
            mt += mt_
            if moved:
                fixations.append(elements[e+1])
        return mt, fixations

    # Given a starting element, exhaustively visually search through
    # all visual elements and return the scanpath and total eye
    # movement time. Next target is always the closest one, and there
    # is perfect inhibition of return.
    def exhaustive_visual_search(self, start = None, target = None):
        if not start:
            start = self.eye_loc
        searched = [start]
        scanpath = [start]
        mt = 0
        while len(searched) != len(self.elements):
            # Figure out the next target, which is the one closest to
            # the current element, and not yet among the searched and
            # thus inhibited elements.
            dist = None
            new_target = None
            for e in self.elements:
                if e not in searched:
                    d = self.element_distance(start, e)
                    if dist == None or dist > d:
                        new_target = e
                        dist = d
            mt_, moved = self.emma_time(new_target, eye_loc = start)
            mt += mt_
            #print(start, target, mt, moved)
            if moved: scanpath.append(new_target)
            searched.append(new_target)
            if new_target == target:
                break
            if moved: start = new_target
        return mt, scanpath

    ## Make an exhaustive search, with item saliency and potentially
    ## requested feature top-down information taken into account.
    def exhaustive_guided_visual_search(self, start = None, target = None, top_down = None, force_fixation = None):
        if not start:
            start = self.eye_loc
        searched = [start]
        scanpath = [start]
        if start == target:
            mt_, moved = self.emma_time(start, eye_loc = start)
            return mt_, scanpath

        rt_pos = None
        if target in self.ltm_pos:
            rt_pos = self.recall_time(self.ltm_pos[target])

        rt_col = None
        if target in self.ltm_color:
            rt_col = self.recall_time(self.ltm_color[target])

        mt = 0
        while len(searched) != len(self.elements):
            # Figure out the next target, which is the one with
            # largest activation, but not inhibited.
            activation = self.bottom_up_activation(start)

            if rt_col and mt >= rt_col:
                top_down = self.ltm_color_fact[target]

            if top_down:
                top_down_activation = self.top_down_activation(top_down, eye_loc = start)
                for e in top_down_activation:
                    if e in activation:
                        activation[e] += top_down_activation[e]

            for e in searched:
                if e in activation:
                    del activation[e]


            new_target = max(activation.items(), key=lambda x: x[1])[0]
            if rt_pos and mt >= rt_pos:
                new_target = self.closest_element(self.ltm_pos_fact[target], elements = list(activation.keys()))
                #print(new_target)

            mt_, moved = self.emma_time(new_target, eye_loc = start)
            mt += mt_
            #print(start, new_target, moved, mt)
            if moved: scanpath.append(new_target)

            searched.append(new_target)
            if new_target == target:
                break

            if moved or force_fixation: start = new_target
        return mt, scanpath


    def bottom_up_activation(self, eye_loc = None):
        if not eye_loc:
            eye_loc = self.eye_loc
        # First create a visual representation of those items that have colour visible.
        visible_elements = {}
        for e in self.elements:
            d = self.visual_distance(e, eye_loc)
            if (0.104*math.pow(d,2) - 0.95*d) < self.angular_size(max(self.elements[e].x_size, self.elements[e].y_size)):
                visible_elements[e] = self.elements[e]
        # Then iterate bottom-up activation
        activation = {}
        for e in visible_elements:
            activation[e] = 0
            for e2 in visible_elements:
                if not e == e2:
                    if visible_elements[e].color != visible_elements[e2].color:
                        activation[e] += 1/math.sqrt(self.element_distance(e, e2))

        # Put elements without any bottom-up activation as zero-activation
        for e in self.elements:
            if e not in activation:
                activation[e] = 0

        return activation

    # Given a requested feature, return the top-down activation of elements.
    def top_down_activation(self, top_down, eye_loc = None):
        if not eye_loc:
            eye_loc = self.eye_loc
        # First set default 0.5 for unseen features
        activation = {}
        for e in self.elements:
            activation[e] = 0.5
        # Then create a visual representation of those items that have colour visible.
        visible_elements = {}
        for e in self.elements:
            d = self.visual_distance(e, eye_loc)
            #print(e, d, (0.104*math.pow(d,2))-(0.95*d),self.angular_size(max(self.elements[e].x_size, self.elements[e].y_size)))
            if (0.104*math.pow(d,2) - 0.95*d) < self.angular_size(max(self.elements[e].x_size, self.elements[e].y_size)):
                visible_elements[e] = self.elements[e]
        # Then iterate bottom-up activation
        for e in visible_elements:
            activation[e] = 0
            if self.elements[e].color == top_down:
                activation[e] += 1

        # Put elements without any top-down activation as zero-activation
        # for e in self.elements:
        #     if e not in activation:
        #         activation[e] = 0

        return activation

    def total_activation(self, top_down = None, eye_loc = None):
        if not eye_loc:
            eye_loc = self.eye_loc
        activation = self.bottom_up_activation(eye_loc)
        # Multiplier for bottom-up
        for e in activation:
            activation[e] = activation[e]*1.1
        if top_down:
            top_down_activation = self.top_down_activation(top_down, eye_loc = eye_loc)
            for e in top_down_activation:
                activation[e] += 0.45*top_down_activation[e]
        return activation

    def WHo_mt(self, start, target, sigma, k_alpha = 0.12):
        x0 = 0.092
        y0 = 0.0018
        alpha = 0.6
        x_min = 0.006
        x_max = 0.06

        distance = self.element_distance(start,target)

        if distance == 0:
            distance = 0.0000001

        mt = pow((k_alpha * pow(((sigma - y0) / distance),(alpha - 1))), 1 / alpha ) + x0

        return mt


# Given the constants a and b, the distance to the target and its
# width, return the Fitts' law based movement time prediction. Using
# formula from Mackenzie (1992).
def fitts_mt(distance, width, a = 0.230, b = 0.166):
    if distance == 0:
        return a
    else:
        return a + b * math.log(distance/width + 1, 2)

# Test
test_ui = ui()

test_ui.add_element("logo", 10, 10, 10, 10, color = "grey")
test_ui.add_element("logo2", 20, 20, 10, 10, color = "yellow")
test_ui.add_element("search", 100, 10, 100, 10, color = "blue")
test_ui.add_element("button", 50, 300, 25, 20, color = "green")

test_ui.fitts_movement_time(["logo","search","button"])

# test_ui.get_task_coordinates(["logo","search"])

import matplotlib.pyplot as plt
# %matplotlib notebook

# If path is provided as a list of element names, it is also drawn.
def visualise_UI(ui, path = [], show_text = True, show_fixation = False, scanpath = False, annotate = False):

    max_x = 0
    max_y = 0

    plt.close() # close any existing plot
    plt.axes()
    plt.xlim(0, ui.x_size)
    plt.ylim(0, ui.y_size)

    plt.gca().invert_yaxis()
    for e in list(ui.elements.values()):
        if e.x+e.x_size > max_x: max_x = e.x+e.x_size
        if e.y+e.y_size > max_y: max_y = e.y+e.y_size
        rectangle = plt.Rectangle((e.x, e.y), e.x_size, e.y_size, fc = e.color)
        plt.gca().add_patch(rectangle)
        if show_text:
            plt.text(e.x, e.y, e.name)
            if e.data:
                plt.text(e.x, e.y+20, str(e.data))

    for p in range(len(path)-1):
        if isinstance(path[p], str):
            loc1 = ui.elements[path[p]].loc()
            # e1_h = ui.elements[path[p]].y_size
            # e2_h = ui.elements[path[p]].y_size
        else:
            loc1 = path[p]
        if isinstance(path[p+1], str):
            col = "black"
            loc2 = ui.elements[path[p+1]].loc()
        else:
            col = "red"
            loc2 = path[p+1]
            
        plt.plot([loc1[0],loc2[0]], [loc1[1],loc2[1]], color = "black") # , marker = 'o'
        circle = plt.Circle((loc2[0],loc2[1]), 30, fc = col)
        plt.gca().add_patch(circle)
        

        if annotate:
            if p == 0:
                plt.text(loc1[0]+20, loc1[1]+20, str(annotate[p]))
            plt.text(loc2[0]+20, loc2[1]+20, str(annotate[p+1]))

    if scanpath:
        circle_size = max(max_x, max_y) / 20 # make sure fixation circle is large enough
        loc = ui.elements[scanpath[0]].loc()
        circle = plt.Circle((loc[0],loc[1]), circle_size, fc = "red", alpha = 0.2)
        plt.gca().add_patch(circle)
        circle = plt.Circle((loc[0],loc[1]), circle_size+1, fc = "black", fill = False)
        plt.gca().add_patch(circle)

        for p in range(len(scanpath)-1):
            loc1 = ui.elements[scanpath[p]].loc()
            loc2 = ui.elements[scanpath[p+1]].loc()
            e1_h = ui.elements[scanpath[p]].y_size
            e2_h = ui.elements[scanpath[p]].y_size
            plt.plot([loc1[0],loc2[0]], [loc1[1],loc2[1]], color = "red", marker = 'o')
            circle = plt.Circle((loc2[0],loc2[1]), circle_size, fc = "red", alpha = 0.2)
            plt.gca().add_patch(circle)
            circle = plt.Circle((loc2[0],loc2[1]), circle_size+1, fc = "black", fill = False)
            plt.gca().add_patch(circle)

    if show_fixation:
        circle_size = max(max_x, max_y) / 20 # make sure fixation circle is large enough
        e = ui.elements[ui.eye_loc]
        circle = plt.Circle((e.x, e.y), circle_size, fc = "red", alpha = 0.2)
        plt.gca().add_patch(circle)

    plt.axis('scaled')
    plt.show()

def visualise_exhaustive_search(ui, start = None, target = None):
    search = ui.exhaustive_visual_search(start = start, target = target)
    visualise_UI(ui, scanpath = search[1])
    print("Total search time:", search[0])
    print("Total number of fixations:",len(search[1]))

def visualise_exhaustive_guided_search(ui, start = None, target = None, top_down = None, force_fixation = None):
    search = ui.exhaustive_guided_visual_search(start = start, target = target, top_down = top_down, force_fixation = force_fixation)
    visualise_UI(ui, scanpath = search[1])
    print("Total search time:", search[0])
    print("Total number of fixations:",len(search[1]))

big_ui = ui(1920, 1280)
m=8
big_ui = ui(1920, 1280)
big_ui.add_element("e1", 10*m,30*m,30*m,20*m, color ="red")
big_ui.add_element("a2", 10*m,60*m,30*m,20*m, color ="red")
big_ui.add_element("a3", 10*m,90*m,30*m,20*m, color ="red")
big_ui.add_element("a4", 10*m,120*m,30*m,20*m, color ="red")

big_ui.add_element("b1", 80*m,30*m,30*m,20*m, color ="green")
big_ui.add_element("b2", 80*m,60*m,30*m,20*m, color ="green")
big_ui.add_element("b3", 80*m,90*m,30*m,20*m, color ="green")
big_ui.add_element("b4", 80*m,120*m,30*m,20*m, color ="green")

big_ui.add_element("e2", 150*m,30*m,30*m,20*m, color ="blue")
big_ui.add_element("c2", 150*m,60*m,30*m,20*m, color ="blue")
big_ui.add_element("c3", 150*m,90*m,30*m,20*m, color ="blue")
big_ui.add_element("c4", 150*m,120*m,30*m,20*m, color ="blue")

big_ui.add_element("d1", 230*m,30*m,30*m,20*m, color ="grey")
big_ui.add_element("d2", 230*m,60*m,30*m,20*m, color ="grey")
big_ui.add_element("d3", 230*m,90*m,30*m,20*m, color ="grey")
big_ui.add_element("d4", 230*m,120*m,30*m,20*m, color ="black")

big_ui.add_element("t1", 130*m,0*m,30*m,20*m, color ="yellow")
big_ui.add_element("t2", 170*m,0*m,30*m,20*m, color ="yellow")
big_ui.add_element("t3", 210*m,0*m,30*m,20*m, color ="yellow")
big_ui.add_element("e3", 250*m,0*m,30*m,20*m, color ="yellow")

#big_ui.learn_element_pos("a4", "expert")
# big_ui.learn_all_elements("expert")

# #visualise_exhaustive_guided_search(big_ui, target = "d4")
# big_ui.swap_elements("c4","d4")
# visualise_exhaustive_guided_search(big_ui, target = "d4")

# colorbad_ui = ui(1920, 1280)
# colorbad_ui.add_element("a1", 10*m,30*m,30*m,20*m, color ="red")
# colorbad_ui.add_element("a2", 10*m,60*m,30*m,20*m, color ="green")
# colorbad_ui.add_element("a3", 10*m,90*m,30*m,20*m, color ="blue")
# colorbad_ui.add_element("a4", 10*m,120*m,30*m,20*m, color ="grey")

# colorbad_ui.add_element("b1", 80*m,30*m,30*m,20*m, color ="green")
# colorbad_ui.add_element("b2", 80*m,60*m,30*m,20*m, color ="grey")
# colorbad_ui.add_element("b3", 80*m,90*m,30*m,20*m, color ="red")
# colorbad_ui.add_element("b4", 80*m,120*m,30*m,20*m, color ="blue")

# colorbad_ui.add_element("c1", 150*m,30*m,30*m,20*m, color ="blue")
# colorbad_ui.add_element("c2", 150*m,60*m,30*m,20*m, color ="grey")
# colorbad_ui.add_element("c3", 150*m,90*m,30*m,20*m, color ="red")
# colorbad_ui.add_element("c4", 150*m,120*m,30*m,20*m, color ="green")

# colorbad_ui.add_element("d1", 230*m,30*m,30*m,20*m, color ="blue")
# colorbad_ui.add_element("d2", 230*m,60*m,30*m,20*m, color ="grey")
# colorbad_ui.add_element("d3", 230*m,90*m,30*m,20*m, color ="red")
# colorbad_ui.add_element("d4", 230*m,120*m,30*m,20*m, color ="green")

# colorbad_ui.add_element("t1", 130*m,0*m,30*m,20*m, color ="red")
# colorbad_ui.add_element("t2", 170*m,0*m,30*m,20*m, color ="blue")
# colorbad_ui.add_element("t3", 210*m,0*m,30*m,20*m, color ="grey")
# colorbad_ui.add_element("t4", 250*m,0*m,30*m,20*m, color ="green")

# res = []
# for i in range(1000):
#     start = random.choice(list(color_ui.elements.keys()))
#     start = "t1"
#     # target = random.choice(list(color_ui.elements.keys()))
#     # target = "c4"
#     # color = color_ui.elements[target].color
#     res.append(color_ui.exhaustive_guided_visual_search(start = start)[0])
# print(np.mean(res))

# res = []
# for i in range(1000):
#     start = random.choice(list(colorbad_ui.elements.keys()))
#     start = "t1"
#     # target = random.choice(list(colorbad_ui.elements.keys()))
#     # target = "c4"
#     # color = colorbad_ui.elements[target].color
#     res.append(colorbad_ui.exhaustive_guided_visual_search(start = start)[0])
# print(np.mean(res))

# target = "t4"
#visualise_exhaustive_guided_search(color_ui, start = "t1")
#visualise_exhaustive_guided_search(colorbad_ui, start = "t1")

# print(color_ui.exhaustive_guided_visual_search(start = "t2"))
# print(colorbad_ui.exhaustive_guided_visual_search(start = "t2"))

# Define disjunctive small and large layouts
disj_small = ui(1900,1280)
m_size = 12
m_pos = 10
alt_color = "blue"
disj_small.add_element("e1", 20*m_pos, 20*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_small.add_element("e2", 40*m_pos, 80*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_small.add_element("e3", 10*m_pos, 120*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_small.add_element("e4", 70*m_pos, 120*m_pos, 20*m_size, 20*m_size, color = "blue")
disj_small.add_element("e5", 100*m_pos, 50*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_small.add_element("e6", 75*m_pos, 80*m_pos, 20*m_size, 20*m_size, color = "blue")
disj_small.add_element("e7", 70*m_pos, 20*m_pos, 20*m_size, 20*m_size, color = "blue")
disj_small.add_element("e8", 110*m_pos, 90*m_pos, 20*m_size, 20*m_size, color = "red")
disj_small.eye_loc = "e7"

disj_large = ui(1900, 1280)
m_size = 12
m_pos = 10
alt_color = "blue"
alt_color2 = "blue"
disj_large.add_element("e1", 20*m_pos, 20*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_large.add_element("e2", 40*m_pos, 90*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_large.add_element("e3", 10*m_pos, 120*m_pos, 20*m_size, 20*m_size, color = "blue")
disj_large.add_element("e4", 70*m_pos, 120*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_large.add_element("e5", 70*m_pos, 20*m_pos, 20*m_size, 20*m_size, color = alt_color2)
disj_large.add_element("e6", 100*m_pos, 100*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_large.add_element("e7", 80*m_pos, 50*m_pos, 20*m_size, 20*m_size, color = "blue")
disj_large.add_element("e8", 120*m_pos, 130*m_pos, 20*m_size, 20*m_size, color = "red")
disj_large.add_element("e9", 75*m_pos, 80*m_pos, 20*m_size, 20*m_size, color = "blue")
disj_large.add_element("e10", 50*m_pos, 160*m_pos, 20*m_size, 20*m_size, color = "blue")
disj_large.add_element("e11", 110*m_pos, 160*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_large.add_element("e12", 140*m_pos, 100*m_pos, 20*m_size, 20*m_size, color = "blue")
disj_large.add_element("e13", 170*m_pos, 120*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_large.add_element("e14", 30*m_pos, 60*m_pos, 20*m_size, 20*m_size, color = alt_color2)
disj_large.add_element("e15", 120*m_pos, 50*m_pos, 20*m_size, 20*m_size, color = alt_color2)
disj_large.add_element("e16", 120*m_pos, 20*m_pos, 20*m_size, 20*m_size, color = alt_color)
disj_large.eye_loc = "e7"


# visualise_UI(disj_large)

# visualise_exhaustive_guided_search(disj_large, start = "e7", target = "e8", top_down = "red")

# disj_small.modify_element("e2", "color", "red")
# visualise_exhaustive_guided_search(disj_small, start = "e7", target = "e8", top_down = "red")

# my_ui = ui()
# my_ui.add_element("e1", 10, 10, 10, 10)
# my_ui.add_element("e2", 210, 10, 10, 10)
# my_ui.emma_time("e1", "e2")
# visualise_exhaustive_search(my_ui)
# my_ui.add_element("e3", 120, 100, 40, 40)
# visualise_exhaustive_search(my_ui)

# s = 0
# for i in range(1,20):
#     s+=math.pow(i*60+2*3600,-0.5)

#s = math.pow(60,-0.5)+math.pow(120,-0.5)

# test_ui.learn_element_color("button", 5)
# visualise_exhaustive_guided_search(test_ui, target = "button")

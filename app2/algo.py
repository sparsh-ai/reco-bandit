import numpy as np
import scipy
import scipy.stats as stats
from vowpalwabbit import pyvw
import random
import pandas as pd
from itertools import product

# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |User users={} context1={} context2={}\n".format(context["user"], context["context1"], context["context2"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action items={} \n".format(action)
    #Strip the last newline
    return example_string[:-1]
def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if(sum_prob > draw):
            return index, prob
def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context, actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob
def choose_user(users):
    return random.choice(users)
def choose_context1(context1):
    return random.choice(context1)
def choose_context2(context2):
    return random.choice(context2)

class VWCSimulation():
    def __init__(self, vw, ictxt):
        self.vw = vw
        self.users = ictxt['users'].unique().tolist()
        self.contexts1 = ictxt['context1'].unique().tolist()
        self.contexts2 = ictxt['context2'].unique().tolist()
        self.actions = ictxt['items'].unique().tolist()
        self.contextdf = ictxt.copy()
        self.contextdf['cost'] = self.contextdf['reward']*-1
        
    def get_cost(self, context, action):
        return self.contextdf.loc[(self.contextdf['users']==context['user']) & \
                (self.contextdf.context1==context['context1']) & \
                (self.contextdf.context2==context['context2']) & \
                (self.contextdf['items']==action), \
                'cost'].values[0]
    
    def update_context(self, new_ctxt):
        self.contextdf = new_ctxt.copy()
        self.contextdf['cost'] = self.contextdf['reward']*-1
    
    def step(self):
        user = choose_user(self.users)
        context1 = choose_context1(self.contexts1)
        context2 = choose_context2(self.contexts2)
        context = {'user': user, 'context1': context1, 'context2': context2}
        action, prob = get_action(self.vw, context, self.actions)
        cost = self.get_cost(context, action)
        vw_format = self.vw.parse(to_vw_example_format(context, self.actions, (action, cost, prob)), pyvw.vw.lContextualBandit)
        self.vw.learn(vw_format)
        self.vw.finish_example(vw_format)
        return (context['user'], context['context1'], context['context2'], action, cost, prob)
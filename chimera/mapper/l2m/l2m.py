#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import chimera
from chimera.mapper import Mapper
import networkx as nx
import time
import json
import openai

class L2M(Mapper):
    def __init__(self, config, device=0, batch_size=1, graph=None, **kwargs):
        self.graph = graph
        if self.graph == None:
            self.graph = nx.Graph()

    def reset(self):
        self.graph = nx.Graph()

    def Right_checker(self, message):
        functions = {
            "name": "Extractor",
            "description": "The function extracts passing waypoints and right-turns among them, if any. If not, enter \"None\" is returned.",
            "parameters": {
                "type":"object",
                "properties":{
                    "waypoints": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "Information about waypoints in the description of the navigation path. Extract information without excesses or deficiencies."
                    },
                    "turn_right_points": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "The point where you turn right in the navigation. If there is none, enter \"None\" is returned."
                    }
                }
            },
            "required": ["waypoints", "turn_right_points"]
        }
        makura = "From the path described below, extract the waypoints. Then, indicate which waypoint right-turns, if any. If there is no left turn, enter \"None\".\"\"\"Path:"
        conmes = makura+message
        inputmessage = {
            "role":"user",
            "content":conmes
        }
        llm = chimera.create_llm(verbose=False, name="gpt-4-1106-preview")
        llm.reset()
        llm.add_function(functions)
        response = llm.chat(inputmessage)
        print("cool down")
        time.sleep(20)
        
        return response

    def Left_checker(self, message):
        functions = {
            "name": "Extractor",
            "description": "The function extracts passing waypoints and left-turns among them, if any. If not, enter \"None\" is returned.",
            "parameters": {
                "type":"object",
                "properties":{
                    "waypoints": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "Information about waypoints in the description of the navigation path. Extract information without excesses or deficiencies."
                    },
                    "turn_left_points": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "Which the point that you turn left in the navigation. If there is none, enter \"None\" is returned."
                    }
                }
            },
            "required": ["waypoints", "turn_left_points"]
        }
        makura = "From the path described below, extract the waypoints. Then, indicate which waypoint left-turns, if any. If there is no left turn, enter \"None\".\"\"\"Path:"
        conmes = makura+message
        inputmessage = {
            "role":"user",
            "content":conmes
        }
        llm = chimera.create_llm(verbose=False, name="gpt-4-1106-preview")
        
        llm.reset()
        llm.add_function(functions)
        response = llm.chat(inputmessage)
        print("cool down")
        time.sleep(20)
        
        return response
        
    def LR_judge(self, message, tpoint):
        functions = {
            "name": "Judgement",
            "description": "This function memorizes the action at a given point, whether Turn right or Turn left.",
            "parameters":{
                "type": "object",
                "properties":{
                    "LR": {
                        "type": "string",
                        "enum": ["Turn left", "Turn right"],
                        "description": "Information about the action at a given point, whether Turn right or Turn left."
                    },
                },
                "required": ["LR"]
            }
        }
        # makura1 = "Function to determine whether the action at point "
        # makura2 = " on the following path is Turn right or Turn left. \"\"\"Path: "
        makura1 = "For the following path, answer whether the action at point "
        makura2 = " is Turn right or Turn left. Using the function. \"\"\"Path: "
        _message = makura1+tpoint+makura2+message
        # print(_message)
        inputmessage = {
            "role":"user",
            "content":_message
        }
        llm = chimera.create_llm(verbose=False)#, name="gpt-4-1106-preview")
        
        llm.reset()
        llm.add_function(functions)
        response = llm.chat(inputmessage)
        print("cool down")
        time.sleep(20)
        
        return response

    def same_checker(self, point, e1, e2, G):
        checker = 0
        for i in range(len(dict(G.nodes)[point])):
            if e1==dict(G.nodes)[point][i][0] and e2==dict(G.nodes)[point][i][1]:
                checker = 1
        return checker

    def add(self, message):

        G = self.graph

        # Convert Message to Map information
        kuri = 0
        knum = 0
        pass_checker = 0

        # Cheking the left/right turning points using prompt
        while kuri==0 and knum<10:
            tr_points = self.Right_checker(message)
            _tr_message = tr_points.tool_calls[0].function.arguments
            tr_message = json.loads(_tr_message)
            tl_points = self.Left_checker(message)
            _tl_message = tl_points.tool_calls[0].function.arguments
            tl_message = json.loads(_tl_message)
            if "turn_right_points" in tr_message:
                if tr_message["turn_right_points"]!=[] and tr_message["turn_right_points"] is not None:
                    tr_ignore = len(tr_message["turn_right_points"])*[0]
            if "turn_left_points" in tl_message:
                if tl_message["turn_left_points"]!=[] and tl_message["turn_left_points"]!=[] is not None:
                    tl_ignore = len(tl_message["turn_left_points"])*[0]
            if "turn_right_points" in tr_message and "turn_left_points" in tl_message:
                if tr_message["turn_right_points"]!=[] and tl_message["turn_left_points"]!=[] and tr_message["turn_right_points"] is not None and tl_message["turn_left_points"]!=[] is not None:
                    for check_tr in range(len(tr_message["turn_right_points"])):
                        for check_tl in range(len(tl_message["turn_left_points"])):
                            if tr_message["turn_right_points"][check_tr]==tl_message["turn_left_points"][check_tl]:
                                judge = self.LR_judge(message, tr_message["turn_right_points"][check_tr])
                                _judge_message = judge.tool_calls.function.arguments
                                judge_message = json.loads(_judge_message)                        
                                if judge_message["LR"]=="Turn left":
                                    tr_ignore[check_tr] = 1
                                else:
                                    tl_ignore[check_tl] = 1
            if tr_message["waypoints"]==tl_message["waypoints"]:
                kuri = 1
            knum += 1
        if knum==10:
            print("Missed_extract_waypoint!")
        else:
            # Making graph
            tr_ex = 0
            tl_ex = 0
            if "turn_right_points" in tr_message:
                if tr_message["turn_right_points"]!=[]:
                    tr_ex = 1
                    tr_num = len(tr_message["turn_right_points"])
                    tr_counter = 0
            if "turn_left_points" in tl_message:
                if tl_message["turn_left_points"]!=[]:
                    tl_ex = 1
                    tl_num = len(tl_message["turn_left_points"])
                    tl_counter = 0
            for f_num in range(len(tr_message["waypoints"])-1):
                straight_checker = 0
                if f_num==0:
                    G.add_node(tr_message["waypoints"][f_num])
                    G.add_node(tr_message["waypoints"][f_num+1])
                    G.add_edge(tr_message["waypoints"][f_num], tr_message["waypoints"][f_num+1])
                else:
                    G.add_node(tr_message["waypoints"][f_num+1])
                    G.add_edge(tr_message["waypoints"][f_num], tr_message["waypoints"][f_num+1])
                    if tr_ex==1 and tr_message["waypoints"][f_num]==tr_message["turn_right_points"][tr_counter]:
                        if tr_ignore[tr_counter]==0:
                            straight_checker = 1
                            pass_checker = self.same_checker(tr_message["waypoints"][f_num], tr_message["waypoints"][f_num-1], tr_message["waypoints"][f_num+1], G)
                            if pass_checker==0:
                                G.add_nodes_from([(tr_message["waypoints"][f_num], {len(dict(G.nodes[tr_message["waypoints"][f_num]])):[tr_message["waypoints"][f_num-1], tr_message["waypoints"][f_num+1], "r"]})])
                            pass_checker = 0
                        if tr_counter==tr_num-1:
                            tr_ex = 0
                        else:
                            tr_counter+=1
                    if tl_ex==1 and tl_message["waypoints"][f_num]==tl_message["turn_left_points"][tl_counter]:
                        if tl_ignore[tl_counter]==0:
                            straight_checker = 1
                            pass_checker = self.same_checker(tr_message["waypoints"][f_num], tr_message["waypoints"][f_num-1], tr_message["waypoints"][f_num+1], G)
                            if pass_checker==0:
                                G.add_nodes_from([(tr_message["waypoints"][f_num], {len(dict(G.nodes[tr_message["waypoints"][f_num]])):[tr_message["waypoints"][f_num-1], tr_message["waypoints"][f_num+1], "l"]})])
                            pass_checker = 0
                        if tl_counter==tl_num-1:
                            tl_ex = 0
                        else:
                            tl_counter+=1
                    if straight_checker==0:
                            pass_checker = self.same_checker(tr_message["waypoints"][f_num], tr_message["waypoints"][f_num-1], tr_message["waypoints"][f_num+1], G)
                            if pass_checker==0:
                                G.add_nodes_from([(tr_message["waypoints"][f_num], {len(dict(G.nodes[tr_message["waypoints"][f_num]])):[tr_message["waypoints"][f_num-1], tr_message["waypoints"][f_num+1], "s"]})])
                            pass_checker = 0
                            
        print(message)

        self.graph = G        

        self.action_estimation()

        outputs = {
            "graph": self.graph,
        }

        return outputs


    def action_estimation(self):
        G = self.graph
        update = True
        while update:
            update = False
            pass_checker = 0
            for Gname in dict(G.nodes).keys():
                for i in range(len(dict(G.nodes)[Gname])):
                    # for j in range(len(dict(G.nodes)[Gname])-1-i):
                    for j in range(i):
                        if dict(G.nodes)[Gname][i][1]==dict(G.nodes)[Gname][j][0]:
                            if dict(G.nodes)[Gname][i][2]=='r' and dict(G.nodes)[Gname][j][2]=='r':
                                pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][i][0], dict(G.nodes)[Gname][j][1], G)
                                if pass_checker==0:
                                    G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][i][0], dict(G.nodes)[Gname][j][1], 's']})])
                                    update = True
                                pass_checker = 0
                            elif dict(G.nodes)[Gname][i][2]=='l' and dict(G.nodes)[Gname][j][2]=='l':
                                pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][i][0], dict(G.nodes)[Gname][j][1], G)
                                if pass_checker==0:
                                    G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][i][0], dict(G.nodes)[Gname][j][1], 's']})])
                                    update = True
                                pass_checker = 0
                            elif dict(G.nodes)[Gname][i][2]=='l' and dict(G.nodes)[Gname][j][2]=='s':
                                pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][i][0], dict(G.nodes)[Gname][j][1], G)
                                if pass_checker==0:
                                    G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][i][0], dict(G.nodes)[Gname][j][1], 'r']})])
                                    update = True
                                pass_checker = 0
                            elif dict(G.nodes)[Gname][i][2]=='r' and dict(G.nodes)[Gname][j][2]=='s':
                                pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][i][0], dict(G.nodes)[Gname][j][1], G)
                                if pass_checker==0:
                                    G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][i][0], dict(G.nodes)[Gname][j][1], 'l']})])
                                    update = True
                                pass_checker = 0
                        elif dict(G.nodes)[Gname][j][1]==dict(G.nodes)[Gname][i][0]:
                            if dict(G.nodes)[Gname][i][2]=='r' and dict(G.nodes)[Gname][j][2]=='r':
                                pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][j][0], dict(G.nodes)[Gname][i][1], G)
                                if pass_checker==0:
                                    G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][j][0], dict(G.nodes)[Gname][i][1], 's']})])
                                    update = True
                                pass_checker = 0
                            elif dict(G.nodes)[Gname][i][2]=='l' and dict(G.nodes)[Gname][j][2]=='l':
                                pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][j][0], dict(G.nodes)[Gname][i][1], G)
                                if pass_checker==0:
                                    G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][j][0], dict(G.nodes)[Gname][i][1], 's']})])
                                pass_checker = 0
                            elif dict(G.nodes)[Gname][j][2]=='l' and dict(G.nodes)[Gname][i][2]=='s':
                                pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][j][0], dict(G.nodes)[Gname][i][1], G)
                                if pass_checker==0:
                                    G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][j][0], dict(G.nodes)[Gname][i][1], 'r']})])
                                    update = True
                                pass_checker = 0
                            elif dict(G.nodes)[Gname][j][2]=='r' and dict(G.nodes)[Gname][i][2]=='s':
                                pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][j][0], dict(G.nodes)[Gname][i][1], G)
                                if pass_checker==0:
                                    G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][j][0], dict(G.nodes)[Gname][i][1], 'l']})])
                                    update = True
                                pass_checker = 0
                for i in range(len(dict(G.nodes)[Gname])):
                    pass_checker = self.same_checker(Gname, dict(G.nodes)[Gname][i][1], dict(G.nodes)[Gname][i][0], G)
                    if pass_checker==0:
                        if dict(G.nodes)[Gname][i][2]=='s':
                            G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][i][1], dict(G.nodes)[Gname][i][0], 's']})])
                            update = True
                        elif dict(G.nodes)[Gname][i][2]=='r':
                            G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][i][1], dict(G.nodes)[Gname][i][0], 'l']})])
                            update = True
                        elif dict(G.nodes)[Gname][i][2]=='l':
                            G.add_nodes_from([(Gname, {len(dict(G.nodes)[Gname]):[dict(G.nodes)[Gname][i][1], dict(G.nodes)[Gname][i][0], 'r']})])
                            update = True
                    pass_checker = 0

        self.graph = G

    def CaToMe(self, canonical):

        # Convert Canonical information to Message
        _canonical = []
        for i in range(len(canonical)):
            if canonical[i][0]=='d':
                _temp = 'Depart from the '
                _mes = _temp + canonical[i][1]
                _canonical.append(_mes)
            elif canonical[i][0]=='a':
                _temp = 'Arrive at the '
                _mes = _temp + canonical[i][2]
                _canonical.append(_mes)
            elif canonical[i][0]=='l':
                _temp = 'Turn left at the '
                _mes = _temp + canonical[i][1]
                _canonical.append(_mes)
            elif canonical[i][0]=='r':
                _temp = 'Turn right at the '
                _mes = _temp + canonical[i][1]
                _canonical.append(_mes)
            elif canonical[i][0]=='t':
                _canonical.append('Turn around')
            elif canonical[i][0]=='s':
                _temp1 = 'Advance from '
                _temp2 = ' to '
                _mes = _temp1 + canonical[i][1] + _temp2 + canonical[i][2]
                _canonical.append(_mes)
        
        _canonical_message = ','.join(_canonical)
        
        systemmessage = {
            "role":"system",
            "content":"Join the following sentences together to form a natural sentence",
        }
        canonical_message = {
            "role":"user",
            "content":_canonical_message,
        }
        llm = chimera.create_llm(verbose=False, name="gpt-4-1106-preview")
        llm.reset()
        llm.add_message(systemmessage)
        response = llm.chat(canonical_message)
        print("cool down")
        time.sleep(20)

        return response

    def get_path(self, start, goal, return_action=True):

        G = self.graph
        if G.has_node(start):
            if G.has_node(goal):
                if nx.has_path(G, source=start, target=goal):
                    path = nx.shortest_path(G, source=start, target=goal)
                else:
                    print("No_Path!")
                    path = None
            else:
                print("No_Node!")
                path = None

        if return_action and path!=None:
            action_list = ['d']
            for i in range(len(path)-2):
                action_cheker = 1
                for j in range(len(dict(G.nodes)[path[i+1]])):
                    if dict(G.nodes)[path[i+1]][j][0]==path[i] and dict(G.nodes)[path[i+1]][j][1]==path[i+2]:
                        action_cheker = 0
                        action_list.append(dict(G.nodes)[path[i+1]][j][2])
                if action_cheker==1:
                    action_list.append('n')
                    print("No_action!")
            action_list.append('a')

        canonical = []
        for i in range(len(path) - 1):
            if action_list[i] == "l" or action_list[i] == "r":
                canonical.append([action_list[i], path[i], 0])
                canonical.append(["s", path[i], path[i+1]])
            else:
                canonical.append([action_list[i], path[i], path[i+1]])
        canonical.append(["a", path[-2], path[-1]])

        res = self.CaToMe(canonical)

        outputs = {
            "path": path,
            "action": action_list,
            "message": res.content,
        }

        return outputs



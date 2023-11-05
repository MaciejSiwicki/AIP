import pygame
from vector import Vector
from constants import *


class Node(object):
    def __init__(self, x, y):
        self.position = Vector(x, y)
        self.neighbors = {UP: None, DOWN: None, LEFT: None, RIGHT: None}

    def render(self, screen):
        for n in self.neighbors.keys():
            if self.neighbors[n] is not None:
                line_start = self.position.asTuple()
                line_end = self.neighbors[n].position.asTuple()  # type: ignore
                pygame.draw.line(screen, WHITE, line_start, line_end, 4)
                pygame.draw.circle(screen, RED, self.position.asInt(), 12)


class NodeGroup(object):
    def __init__(self):
        self.nodeList = []

    def setupTestNodes(self):
        nodeA = Node(80, 80)
        nodeB = Node(160, 80)
        nodeC = Node(80, 160)
        nodeD = Node(160, 160)
        nodeE = Node(208, 160)
        nodeF = Node(80, 320)
        nodeG = Node(208, 320)
        nodeA.neighbors[RIGHT] = nodeB  # type: ignore
        nodeA.neighbors[DOWN] = nodeC  # type: ignore
        nodeB.neighbors[LEFT] = nodeA  # type: ignore
        nodeB.neighbors[DOWN] = nodeD  # type: ignore
        nodeC.neighbors[UP] = nodeA  # type: ignore
        nodeC.neighbors[RIGHT] = nodeD  # type: ignore
        nodeC.neighbors[DOWN] = nodeF  # type: ignore
        nodeD.neighbors[UP] = nodeB  # type: ignore
        nodeD.neighbors[LEFT] = nodeC  # type: ignore
        nodeD.neighbors[RIGHT] = nodeE  # type: ignore
        nodeE.neighbors[LEFT] = nodeD  # type: ignore
        nodeE.neighbors[DOWN] = nodeG  # type: ignore
        nodeF.neighbors[UP] = nodeC  # type: ignore
        nodeF.neighbors[RIGHT] = nodeG  # type: ignore
        nodeG.neighbors[UP] = nodeE  # type: ignore
        nodeG.neighbors[LEFT] = nodeF  # type: ignore
        self.nodeList = [nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG]

    def render(self, screen):
        for node in self.nodeList:
            node.render(screen)

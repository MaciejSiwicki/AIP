import pygame
from pygame.locals import QUIT
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from text import TextGroup
from sprites import MazeSprites


class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.clock = pygame.time.Clock()
        self.lives = 1
        self.score = 0
        self.textgroup = TextGroup()

    def action(self, action):
        if self.pacman.alive:
            if action == -2:  # Assuming -2 means move right
                self.pacman.direction = RIGHT
            elif action == 2:  # Assuming 2 means move left
                self.pacman.direction = LEFT
            elif action == 1:  # Assuming 1 means move up
                self.pacman.direction = UP
            elif action == -1:  # Assuming -1 means move down
                self.pacman.direction = DOWN
            elif action == 0:  # Assuming 0 means stop
                self.pacman.direction = STOP

    def evaluate(self):
        reward = 0
        if self.lives == 0:
            reward = (
                -self.pellets.getPelletSize()
                - self.pellets.numEaten
                + self.pellet.numEaten
            )
            return reward
        elif self.pellets.isEmpty():
            reward = 100 + self.pellets.numEaten
        elif self.pacman.alive:
            reward = self.pellets.numEaten

        # print(reward)
        return reward

    def is_done(self):
        if self.lives == 0 or (self.pellets.isEmpty() and self.pacman.alive):
            return True
        else:
            return False

    def observe(self):
        # print(
        # self.pacman.position,
        # self.ghosts.blinky.position,
        # self.ghosts.blinky.direction,
        # self.ghosts.blinky.mode.current,
        # self.ghosts.pinky.position,
        # self.ghosts.pinky.direction,
        # self.ghosts.pinky.mode.current,
        # self.ghosts.inky.position,
        # self.ghosts.inky.direction,
        # self.ghosts.inky.mode.current,
        # self.ghosts.clyde.position,
        # self.ghosts.clyde.direction,
        # self.ghosts.clyde.mode.current,
        # self.pellets.numEaten,
        # self.pellets.powerpellets
        return (
            self.pacman.position,
            self.ghosts.blinky.position,
            self.ghosts.blinky.direction,
            self.ghosts.blinky.mode.current,
            self.ghosts.pinky.position,
            self.ghosts.pinky.direction,
            self.ghosts.pinky.mode.current,
            self.ghosts.inky.position,
            self.ghosts.inky.direction,
            self.ghosts.inky.mode.current,
            self.ghosts.clyde.position,
            self.ghosts.clyde.direction,
            self.ghosts.clyde.mode.current,
            self.pellets.numEaten,
            # self.pellets.powerpellets
        )

    def restartGame(self):
        self.lives = 1
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def startGame(self):
        self.setBackground()
        self.mazesprites = MazeSprites(
            "utils/maze1.txt",
            "utils/maze1_rotation.txt",
        )
        self.background = self.mazesprites.constructBackground(self.background, 0)
        self.nodes = NodeGroup("utils/maze1.txt")
        self.nodes.setPortalPair((0, 17), (27, 17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12, 14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15, 14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("utils/maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2 + 11.5, 0 + 14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0 + 11.5, 3 + 14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4 + 11.5, 3 + 14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)
        return self

    def update(self):
        dt = self.clock.tick(30) / 1000.0
        self.textgroup.update(dt)
        self.pacman.update(dt)
        self.ghosts.update(dt)
        self.pellets.update(dt)
        self.checkPelletEvents()
        self.checkGhostEvents()
        self.checkEvents()
        self.render()

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.updateScore(ghost.points)
                    self.textgroup.addText(
                        str(ghost.points),
                        WHITE,
                        ghost.position.x,
                        ghost.position.y,
                        8,
                        time=1,
                    )
                    self.ghosts.updatePoints()
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.pacman.die()
                        if self.lives <= 0:
                            self.restartGame()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

    def render(self):
        self.screen.blit(self.background, (0, 0))  # type: ignore
        self.pellets.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        pygame.display.update()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            # print(self.pellets.numEaten)
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.restartGame()


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()

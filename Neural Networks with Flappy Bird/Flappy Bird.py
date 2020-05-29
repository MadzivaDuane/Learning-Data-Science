#Flappy Bird AI
#install all necessary packages
import pygame
pygame.font.init(). #for pygame fonts 
import random 
import time 
import os
import neat 
from dask import visualize
import sys

#set game screen width and height
win_width = 550
win_height = 800
floor = 730
WIN = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Flappy Bird")
gen = 0
#load images 
#load the birds first - three birds are in motion of flapping wings
#transform.scale2x - increases size of image by 2, image.load - loads the image
path = "/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Tech With Tim"  #directory
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join(path+"/imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join(path+"/imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join(path+"/imgs", "bird3.png")))] #create them as a list 
#pipe, base and background images
pipe_image = pygame.transform.scale2x(pygame.image.load(os.path.join(path+"/imgs", "pipe.png")))
base_image = pygame.transform.scale2x(pygame.image.load(os.path.join(path+"/imgs", "base.png")))
background_image = pygame.transform.scale2x(pygame.image.load(os.path.join(path+"/imgs", "bg.png")))

#create score bar
stat_font = pygame.font.SysFont("comicsans", 50)
end_font = pygame.font.SysFont("comicsans", 70)
draw_lines = False
#create classes to define bird movement 
class Bird:
    images = bird_images
    #set some constants on the bird
    max_rotation = 25 #25 degrees is the maximal tilt the bird can have as it moves
    rot_vel = 20 #rotation velocity
    animation_time = 5  #how fast a frame moves, essentially how quick the bird wings flap

    #define init class
    def __init__(self, x, y):   #x, y are positions of the bird
        self.x = x
        self.y = y
        self.tilt = 0  #where the birds tilt starts
        self.tick_count = 0
        self.vel = 0 #bird velocity at start
        self.height = self.y #where the birds height starts
        self.img_count = 0
        self.img = self.images[0]   #starts of at image 1

    #define the jumping class
    def jump(self):
        self.vel = -10.5 #the - is because the pygames coordinate system is oriented such that the top left corner is (0,0)
        self.tick_count = 0 #helps us keep track of our last jump
        self.height = self.y

    #define move class
    def move(self):
        self.tick_count += 1
        d = self.vel*self.tick_count + 1.5*self.tick_count**2  #displacement (d) from newtons laws of motion

        #now to failsafe this, and ensure the velocity does not overshoot
        if d >= 16:
             d = (d/abs(d))*16

        if d < 0:
            d -=2   #if we are moving upwards, lets move up a little more (jump)
        
        self.y = self.y + d   
        #configure upward tilt and set maximum tilt 
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation
            
        else: 
            if self.tilt > -90:
                self.tilt -= self.rot_vel
    
    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.animation_time:
            self.img = self.images[0]
        elif self.img_count < self.animation_time*2:
            self.img = self.images[1]
        elif self.img_count < self.animation_time*3:
            self.img = self.images[2]
        elif self.img_count < self.animation_time*4:
            self.img = self.images[1]
        elif self.img_count < self.animation_time*4 + 1:
            self.img = self.images[0]
            self.img_count = 0 
        #when the bird is falling, we dont want it to be flapping its wings so we change pick the level winged bird for the fall
        if self.tilt <= -80:
            self.img = self.images[0]
            self.img_count = self.animation_time*2 #so when the bird starts going up it doesnt look akward
        #create a function to rotate the image around its center
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)  #blit just means draw

    def get_mask(self): #for collisions
        return pygame.mask.from_surface(self.img)
#we need to install objects into the birds cours
#start with pipes
class Pipe(): #y excluded because height remains relatively the same 
    pipe_gap = 200 #gap between the 2 pipes
    vel = 5  #velocity of pipes

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.pipe_top = pygame.transform.flip(pipe_image, False, True)
        self.pipe_bottom = pipe_image

        self.passed = False
        self.set_height()
    
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.pipe_top.get_height()
        self.bottom = self.height + self.pipe_gap 
    
    #set class to move the pipe
    def move(self):
        self.x -= self.vel  #moves pipes to the left
    
    #define the drawing of the pipe
    def draw(self, win):
        win.blit(self.pipe_top, (self.x, self.top))  #top pipe 
        win.blit(self.pipe_bottom, (self.x, self.bottom))  #bottom pipe

    #defining a collision uwing pixels- when user flies into pipes or the ground
    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.pipe_top)
        bottom_mask = pygame.mask.from_surface(self.pipe_bottom)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        
        return False

#create the floor of the game
class Base():
    vel = 5
    width = base_image.get_width()
    img = base_image

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel
        #this essentially creates 2 base images and cycles between the two as one completes a cycle on the screen
        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width
    
    def draw(self, win):
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))

#create a function to draw the window of play -  background and bird
def draw_window(win, birds, pipes, base, score, gen, pipe_ind):
    if gen == 0:
        gen = 1

    win.blit(background_image, (0,0)) #draws the background image in 

    for pipe in pipes:
        pipe.draw(win)  #draw in pipes 
    
    base.draw(win)   #draw in base

    for bird in birds:
        # draw lines from bird to pipe
        if draw_lines:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].pipe_top.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].pipe_bottom.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        # draw bird
        bird.draw(win)

    # score
    score_label = stat_font.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (win_width - score_label.get_width() - 15, 10))

    # generations
    score_label = stat_font.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = stat_font.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))

    pygame.display.update()

#create a function to loop in the game
def main(genomes, config):

    global WIN, gen
    win = WIN
    gen += 1

    birds = []  #initialize, starting position
    nets = []  #list of neral networks
    ge = [] #list of genomes

    for genome_id, g in genomes:
        g.fitness = 0  #set initial fitness to 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))  #append bird object to birds list
        ge.append(g)

    base = Base(730) #initialize base
    pipes = [Pipe(700)]  #initialize pipes at height 700
    clock = pygame.time.Clock()
    score = 0 #start the score at 0
    
    run = True
    #create a pathway to quit the game
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                sys.exit()
                break
        #add movement for the bird
        #bird.move()
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipe_top.get_width():
                pipe_ind = 1
        else:
            run = False
            break
        
        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1 #give an initial fitness for bird survival 
            bird.move()  #get each bird to move
            #now to set up a nearal network for each bird
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()

        base.move()  #starts moving base

        add_pipe = False
        remove_pipe = []
        for pipe in pipes:  #move more than one pipe
            pipe.move()

            for bird in birds:
                if pipe.collide(bird, win):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))
            
            if pipe.x + pipe.pipe_top.get_width() < 0: #if pipe is off the screen
                remove_pipe.append(pipe)
            
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1 #if a pipe is passed/ added, increase the score by 1
            for g in ge: #add fitness score for each pipe a bird passes, to encourage them to pass thru pipes rather than ramming past
                g.fitness += 5
            pipes.append(Pipe(win_width))

        for r in remove_pipe:
            pipes.remove(r)

        for bird in birds:  #check if bird hits the ground
            if bird.y + bird.img.get_height() - 10 >= floor or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))
  
        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)
    

"""Utilizing the NEAT Module for AI: https://neat-python.readthedocs.io/en/latest/
In order to use the NEAT module for Flappy Bird, we need to think of our neural network in terms of:
1. Input: Bird y position (since it doesn't move on the x-axis), Bird Position from top and bottom pipe (for collision)
2. Output: Jump (move up or down)
3. Activation function: TanH/ Sigmoidal - reduces values to between -1 and 1, so we can decide at which probability can we jump or not jump
4. Population size - the number of birds we will train per session, can be 10/100/1000 etc, the successful birds are then mutated to produce a new geenration of birds which are tested until we have a perfect bird
5. Fitness function: method to evaluate how good the birds are, in this, case how many frames away from the starting point a bird is
6. Max generation:  iterations of the AI until a bird can be found
"""

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)  #sets population size as stipilated in configuration file

    p.add_reporter(neat.StdOutReporter(True))  #gives report on perfomance
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)   #calls the main function 50 times, evaluating the best performing bird

    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

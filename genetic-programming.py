import math
import sys
from copy import deepcopy
import random


MAX_DEPTH = 4
POP_SIZE = 100
KEEP_COUNT = 20
DROP_COUNT = 20

OPS = ['*','+','-','**']

#
#
#
class Chromosome():
  
  def __init__(self, tree):
    self.tree = tree
    self.ensure_xs_exist()
    self.fitness_score=-1.0
  
  def mutate(self):
    mutate_index = random.randrange(0, self.node_count, 1)
    mutation = self.get_subtree(mutate_index)
    if mutation is None:
      return
    if mutation.type == 'o':
      mutation.val = random.sample(OPS, 1)[0]
    else:
      if mutation.val != 'x':
        mutation.val = str(self.close_int(int(mutation.val)))
    
    
#     again = True
#     count = 0
#     while again and count < 5:
#       count+=1
#       mutate_index = random.randrange(0, self.node_count, 1)
#       mutation = self.get_subtree(mutate_index)
#       if mutation is None:
#         return
#       if mutation.type == 'v':
#         if mutation.val != 'x':
#           mutation.val = str(self.close_int(int(mutation.val)))
#           again = False
#       else:
#         again = True
  
  #
  #  Returns an int no more than -2 to +2 relative to the parameter.  Uses gaussian distr. mean = 0, stdev=0.8 to determine 
  #  the change.
  #    
  def close_int(self, val):
    gaussian = random.gauss(0, 0.8)
    if gaussian<-1:
      return val-2
    elif gaussian < -0.05:
      return val-1
    elif gaussian < 0.05:
      return val
    elif gaussian < 1:
      return val+1
    else:
      return val+2
  #
  #  Adds either 1 or 2 variables (x) to the equation if there are none.  
  #  Also assigns indices and counts the nodes.
  #  
  def ensure_xs_exist(self):
    x_found = False
    queue = []
    queue.append(self.tree)
    index = 0
    while(len(queue)>0):
      node=queue.pop(0)
      node.index = index
      index += 1
      if node.val == 'x':
        x_found = True
      if node.left_child is not None:
        queue.append(node.left_child)
      if node.right_child is not None:
        queue.append(node.right_child)
    self.node_count = index
    
    if x_found:
      return
    else:
      xs_to_add = random.randrange(1,3,1)
      xs_added = 0
      queue = []
      queue.append(self.tree)
      nodes_left = self.node_count
      while(len(queue)>0):
        node=queue.pop(0)
        rand_val = random.random()
        if node.type == 'v':
#            and (node.parent is not None and node.parent.val != '**')
          prob = (1.0 * xs_to_add) / nodes_left
          if rand_val <= prob:
            node.val = 'x'
            xs_to_add -= 1
        nodes_left -= 1
        if node.left_child is not None:
          queue.append(node.left_child)
        if node.right_child is not None:
          queue.append(node.right_child)
      
  #
  #  Returns the subtree at location specified by retr_val
  #
  def get_subtree(self, retr_val):
    index = 0
    queue = []
    queue.append(self.tree)
    while(len(queue)>0):
      node=queue.pop(0)
      node.index = index
      if retr_val == node.index:
        return node
      index += 1
      if node.left_child is not None:
        queue.append(node.left_child)
      if node.right_child is not None:
        queue.append(node.right_child)
        
  #
  #  Generates a string representation of the expression
  #  represented by a tree of Node objects
  #      
  def expression(self):
    return self.expr_rec(self.tree)
      
  def expr_rec(self, node):  
    if node == None:
      return ''
    
    left = ''
    right = ''
    
    left_child = node.left_child
    
    if left_child is not None:
      if left_child.type == 'v':
        left = left_child.val
      else:
        left = self.expr_rec(left_child)
    
    right_child = node.right_child
    
    if right_child is not None:
      if right_child.type == 'v':
        right = right_child.val
      else:
        right = self.expr_rec(right_child)
    
    return '('+left+node.val+right+')'
  
  #
  #  Returns the variance of the differences of the data and the tree's expression.
  #
  def fitness(self, data):
  
    expression = self.expression()
    total = 0
    
    count = len(data)
    try:
      for coords in data:
        (x, y) = coords
        
        rep_expression = str.replace(expression, 'x', str(x))
        calc_y = 0
        try:
          calc_y = int(eval(rep_expression))
        except OverflowError:
          calc_y = 2320000
        except ZeroDivisionError:
          calc_y = 2320000
        total += int(math.pow(calc_y - y, 2))
    except OverflowError:
      total = 2320000
    
    avg_diff = 1.0*total / count 
    
    rms = int(math.sqrt(avg_diff))
    
    self.fitness_score = rms
    return self.fitness_score
    
    
#
#  Node of an expression tree
#
class Node():
  
  def __init__(self, type, val, parent, depth):
    self.index = 0
    self.type = type
    self.val = val
    self.parent = parent
    self.left_child = None
    self.right_child = None
    self.depth = depth
  
  #
  #  root = True if this is the root from which to calculate height (aka the initial call)
  #
  def calc_depth(self, root):
    right_depth= 0
    left_depth = 0
    if self.left_child is not None:
      left_depth = 1+self.left_child.calc_depth(False)
    if self.right_child is not None:
      right_depth = 1+self.right_child.calc_depth(False)
    
    if root:
      left_depth +=1
      right_depth +=1
    
    return max(left_depth, right_depth)

def generate_random_tree():
  root = generate_single_tree(None, 0, 'm')
  return root
#
#  Generates a random equation in tree form up to MAX_DEPTH in size
#
def generate_single_tree(parent, depth, side):
  
  if (depth>4):
    return None
  depth += 1
  val = ''
  type = 'v'
  type_rand = random.random()
  if (depth != 1 and type_rand <= 0.25) or depth == MAX_DEPTH:
#      or (parent is not None and parent.val == '**' and side == 'r')
    max_range = 20
    if parent.val == '**':
      max_range = 6
    val = str(random.randrange(-1*max_range, max_range, 1))
  elif type_rand < 1 or depth == (MAX_DEPTH-1):
    val = random.sample(OPS,1)[0]
    type = 'o'
  node = Node(type, val, parent, depth)
  if type != 'v':
    node.left_child = generate_single_tree(node, depth, 'l')
    node.right_child = generate_single_tree(node, depth, 'r')
  return node


#
#  Generates its Chromosome's fitness value, as well as calculating the average 
#  fitness value from the top 5 chromosomes.
#  
def assignFitness(in_pop, data):
  for chrom in in_pop:
    chrom.fitness(data)  
    
  sorted_list = sorted(in_pop, key=lambda x: x.fitness_score)
  total = 0.0
  for i in range(0, 5):
    total += sorted_list[i].fitness_score
  avg_fit = (1.0 * total) / 5
  print ("Avg: " + str(avg_fit))
  return (sorted_list, avg_fit)


#
#  Divides the population into two lists.  The highest scoring stay alive, the bottom scoring die off,
#  and the middle selection go on to breed with each other to generate the next population.
#
def play_god(population):
  return (population[:KEEP_COUNT], population[KEEP_COUNT:(len(population) - DROP_COUNT)])
  
def pick_lucky_chromosome(darwins):
  ll = len(darwins)
  gauss = 0.0
  neg = True
  while (neg):
    neg = False
    gauss = random.gauss(0, .5)
    if gauss < 0.0:
      neg = True
  val = int(gauss/2 * ll)
  if val > ll:
    val = ll-1
  return val
#   return random.randrange(0, 60, 1)

#
#  Iterates through the darwins list of Chromosomes randomly selecting two and breeding them.
#  Afterwards it mutates the offspring with a 10% probability.
#
def breed_and_mutate(darwins):
  next_gen = []
  while len(next_gen) < (POP_SIZE-(KEEP_COUNT)):
    first = pick_lucky_chromosome(darwins)
    second = first
    while (second == first):
      second = pick_lucky_chromosome(darwins)
    newborn = breed(darwins[first], darwins[second])
    mutate = random.random()
    if mutate<0.1:
      newborn.mutate()
    next_gen.append(newborn)
  return next_gen

def no_more_subtrees(node):
  return node.left_child is None and node.right_child is None

#
#  Traverses down the two expression trees, randomly picking a node and
#  then mixing the two expression trees at that point.  Will always pick a node 
#  as the probability to pick one raises accordingly to how many nodes
#  are left.
#    
def breed(c1, c2):
  
  rand_dir = random.random()
  curr_dir = 'r'
  
  #  initially traverse the right or left tree from the root with equal odds
  next1 = c1.tree.right_child
  next2 = c1.tree.right_child
  if rand_dir < 0.5:
    next1 = c1.tree.left_child
    next2 = c2.tree.left_child
    curr_dir = 'l'
  
  depth_left = MAX_DEPTH - 1
  while(True):
    rand_fin = random.random()
    prob = 1.0 / depth_left
    if rand_fin <= prob or depth_left==1 or (no_more_subtrees(next1) or no_more_subtrees(next2)):
      sv_p1 = next1.parent
      next1.parent = next2.parent
      next2.parent = sv_p1
      if curr_dir == 'l':
        next1.parent.left_child = next1
        next2.parent.left_child = next2
      else:
        next1.parent.right_child = next1
        next2.parent.right_child = next2
      break
    rand_dir = random.random()
    curr_dir = 'r'
    next1 = c1.tree.right_child
    next2 = c1.tree.right_child
    if rand_dir < 0.5:
      next1 = c1.tree.left_child
      next2 = c2.tree.left_child
      curr_dir = 'l'
      
    depth_left -= 1
  
  cnew = None
  if random.random()<0.5:
    cnew = Chromosome(deepcopy(c1.tree))
  else:
    cnew = Chromosome(deepcopy(c2.tree))
  return cnew



#  ==================================================================================
#  ==================================================================================
#  Main Program (the part that starts to run)
#
#  ==================================================================================

# n0 = Node('o','**',None,1)
# n1 = Node('v','4',n0,2)
# n2 = Node('v','0',n0,2)
# n0.left_child = n1
# n0.right_child = n2
# 
# c0 = Chromosome(n0)
# print (c0.expression())

# exit()


coords = []
# filename = sys.argv[1]
filename = 'resources/fn5.csv'
file = open(filename)
for line in file:
  xy = str.split(line, ',')
  x = int(xy[0])
  y = int(xy[1])
  coords.append((x,y))
  

population = []

#  Generate Initial Population
for i in range(0, POP_SIZE):
  node = generate_random_tree()
  chrom = Chromosome(node)
  population.append(chrom)
population.sort(key=lambda x: x.fitness_score)
(population, avg_fit) = assignFitness(population, coords)


prev_fit = 0.0
again = True
iterations = 0

while again and iterations < 100 and avg_fit > 3:
  iterations += 1
  prev_fit = avg_fit
  (survives, darwins) = play_god(population)
  population.clear()
  population.extend(survives)
  population.extend(breed_and_mutate(darwins))
  (population, avg_fit) = assignFitness(population, coords)
  
  if avg_fit <= prev_fit and (prev_fit - avg_fit < 0.3) and avg_fit < 10:
    again = False
  if population[0].fitness_score - 0 < 0.1:
    again = False
  

for i in range(0, 5):
  print (population[i].expression() + ", fitness="+str(population[i].fitness(coords)))

  
#include "widereach.h"
#include "helper.h"

#include <math.h>
#include <memory.h>

double ucb_score(env_t *env, tnode *child, tnode *parent) {
  double xbar = child->stats.mean_obj;
  int Np = parent->stats.times_visited + 1;
  int Nc = child->stats.times_visited + 1;
  double C = env->params->greer_params.mcts_ucb_const;

  double score = xbar + C*sqrt(log(Np)/Nc);
  //printf("ucb = %g, xbar = %g\n", score, xbar);
  //printf("ucb = %g, xbar = %g, child key = %d, child obj = %g\n", score, xbar, child->key, child->data->res.obj);
  /*if(fabs(score - 254.888) < 1e-3) {
    printf("child key = %d, xbar = %g, Np = %d, Nc = %d\n", child->key, xbar, Np, Nc);
    }*/
  return score;
}

tnode *mcts_select(env_t *env, tnode *root) {
  //greedily chooses the best leaf, according to the UCB scoring method
  //uses a recursive search
  tnode *child = root->l_child;
  if(!child)
    return root;
  double best_score = -1e101;
  tnode *best_child = NULL;
  while(child) {
    double score = ucb_score(env, child, root);
    if(score > best_score) {
      best_score = score;
      best_child = child;
    }
    child = child->r_sib;
  }
  tnode *res = mcts_select(env, best_child);
  //printf("selected node %d, with ucb %g\n", res->key, ucb_score(env, res, res->parent));
  return res;
}

extern int tree_nodes;

tnode *create_random_child(env_t *env, tnode *node) {
  //creates a random child of the given node and returns it
  //does not create an id, branching data, or score
  vsamples_t *vs = env->vsamples;

  if(node->depth == vs->dimension - 1) {
    //printf("node %d is at max depth, no children added\n", node->key);
    return NULL;
  }
  
  int nfp = node->data->res.nfpos, nfn = node->data->res.nfneg;
  int class_rand = rand() % (nfp + nfn);
  sample_locator_t loc;
  if(class_rand >= nfp) {
    //select a false negative
    int class = 1;
    int sample_rand = rand() % nfn;
    loc.class = class, loc.index = node->data->res.fneg[sample_rand];
  } else {
    //select a false positive
    int class = 0;
    int sample_rand = rand() % nfp;
    loc.class = class, loc.index = node->data->res.fpos[sample_rand];
  }
  
  tnode *child = new_empty_child(node);
  child->loc = loc;
  child->depth = node->depth + 1;
  child->data->v = obtain_perp(env, child);
  child->data->res = classify(env, child->data->v);

  tree_nodes++; //this needs to be done, since free_subtree subtracts from it too

  return child;
}

tnode *create_random_child_fast(env_t *env, tnode *node) {
  //creates a random child of the given node and returns it
  //does not create an id, branching data, or score
  vsamples_t *vs = env->vsamples;
  gsl_vector *v = node->data->v;
  gsl_vector *s;
  
  int P = vs->count[1], N = vs->count[0];
  sample_locator_t loc;

  while(1) {
    int class_rand = rand() % (P + N);
    if(class_rand >= N) {
      //select a positive sample
      int class = 1;
      int sample_rand = rand() % P;
      loc.class = class, loc.index = node->data->res.fneg[sample_rand];
    } else {
      //select a negative sample
      int class = 0;
      int sample_rand = rand() % N;
      loc.class = class, loc.index = node->data->res.fpos[sample_rand];
    }
    s = vs->samples[loc.class][loc.index];
    if(dot(v, s) < 0)
      break;
  }
  
  tnode *child = new_empty_child(node);
  child->loc = loc;
  child->depth = node->depth + 1;
  child->data->v = obtain_perp(env, child);

  return child;
}

double min_child_obj(env_t *env, tnode *node) {
  //computes the minimum objective of a node's children
  double min = 1e101;
  tnode *curr = node->l_child;
  while(curr) {
    if(curr->data->res.obj < min) {
      min = curr->data->res.obj;
    }
    curr = curr->r_sib;
  }
  return min;
}

int has_child_loc(tnode *node, sample_locator_t loc) {
  //1 if node has a child with the given loc, 0 otherwise
  tnode *curr = node->l_child;
  while(curr) {
    if(curr->loc.class == loc.class && curr->loc.index == loc.index)
      return 1;
    curr = curr->r_sib;
  }
  return 0;
}

void create_remaining_children(env_t *env, tnode *node) {
  //TODO: this is slow
  //could be made faster by copying node first
  vsamples_t *vs = env->vsamples;
  if(node->depth >= vs->dimension) {
    printf("attempted to add nodes beyond max depth\n");
    return;
  }

  if(node->depth == vs->dimension - 1) {
    return;
  }

  for(size_t i = 0; i < node->data->res.nfneg; i++) {
    //positive samples classified wrong
    sample_locator_t loc = (sample_locator_t) {.class = 1, .index = node->data->res.fneg[i]};
    if(!has_child_loc(node, loc))
      create_node_child(env, node, loc, node->branching_data, compute_score);
  }
  for(size_t i = 0; i < node->data->res.nfpos; i++) {
    //negative samples classified wrong
    sample_locator_t loc = (sample_locator_t) {.class = 0, .index = node->data->res.fpos[i]};
    if(!has_child_loc(node, loc))
      create_node_child(env, node, loc, node->branching_data, compute_score);
  }  
}

tnode *create_best_child(env_t *env, tnode *node) {
  //creates the best child
  //if the node already has children, will compute the best child not currently present
  //requires computing all children, so this is slow

  //TODO: somehow this causes a segfault later on down the line

  //will copy the node and make all children, then take the best one which is not already in the tree
    
  tnode *root = CALLOC(1, tnode);
  memcpy(root, node, sizeof(tnode));
  root->l_sib = root->r_sib = root->l_child = root->r_child = NULL;
  create_node_children_notrim(env, root);

  double best_obj = -1e101;
  tnode *best_child = NULL;
  
  tnode *curr = root->l_child;
  if(!curr) {
    return NULL;
  }
  
  while(curr) {
    if(curr->data->res.obj > best_obj && !has_child_loc(node, curr->loc)) {
      best_obj = curr->data->res.obj;
      best_child = curr;
    }
    curr = curr->r_sib;
  }

  create_node_child(env, node, best_child->loc, node->branching_data, compute_score);

  if(root->l_child)
    free(free_subtree(root->l_child));
  free(root);
  
  return node->r_child;
}

int cmp_obj(const void *x1, const void *x2) {
  return (*(tnode **) x1)->data->res.obj - (*(tnode **) x2)->data->res.obj;
}

tnode *create_best_child1(env_t *env, tnode *node) {
  //creates the best child of the node which doesn't already exist
  if(!node)
    return NULL;
  if(node->depth == env->vsamples->dimension - 1) {
    return NULL;
  }
  tnode *last_existing_child = node->r_child;
  int prev_n_children = node->n_children;
  create_node_children_notrim(env, node);
  int new_children = node->n_children - prev_n_children;
  printf("new_children = %d, prev_n_children = %d\n", new_children, prev_n_children);
  if(new_children <= prev_n_children) {//i.e. all children already exist
    tnode *curr = node->r_child;
    while(curr != last_existing_child) {
      tnode *next = curr->l_sib;
      tree_remove_node(curr);
      free(free_subtree(curr));
      curr = next;
    }
    return NULL;
  }
  tnode **children = CALLOC(new_children, tnode *);

  int k = 0;
  tnode *curr = node->r_child;
  while(curr != last_existing_child) {
    children[k++] = curr;
    curr = curr->l_sib;
  }

  qsort(children, new_children, sizeof(tnode *), cmp_obj);

  tnode *child = children[new_children-1-prev_n_children];

  curr = node->r_child;
  while(curr != last_existing_child) {
    tnode *next = curr->l_sib;
    if(curr != child) {
      tree_remove_node(curr);
      free(free_subtree(curr));
    } 
    curr = next;
  }
  free(children);

  return child;
}

tnode *random_sample_weighted(tnode **items, double *weights, int n) {
  //returns a random item, according to the weights probs
  //weights need not be in [0, 1]; this function will adjust them to be >= 0 by adding the minimum to all, then normalize and consider them as probabilities
  if(n == 0)
    return NULL;
  double min_weight = 1e101;
  double max_weight = -1e101;
  double sum = 0;
  for(int i = 0; i < n; i++) {
    if(weights[i] < min_weight)
      min_weight = weights[i];
    if(weights[i] > max_weight)
      max_weight = weights[i];
  }
  for(int i = 0; i < n; i++)
    sum += weights[i] + min_weight;
  double *cumulative = malloc(n*sizeof(double));
  cumulative[0] = (weights[0] + min_weight)/sum;
  for(int i = 1; i < n; i++) {
    cumulative[i] = cumulative[i-1] + (weights[i] + min_weight)/sum;
  }
  double r = rand() / (double) RAND_MAX;
  int idx;
  for(idx = 0; cumulative[idx] < r; idx++);

  printf("returning an item with weight %g, while max weight = %g\n", weights[idx], max_weight);
  return items[idx];
}

tnode *create_random_child_weighted(env_t *env, tnode *node) {
  //creates a random child of node, weigthed according to the objective
  //this may lead to duplicate children if calleede on a node that already has children
  tnode *last_existing_child = node->r_child;
  int prev_n_children = node->n_children;
  create_node_children_notrim(env, node);
  int new_children = node->n_children - prev_n_children;
  tnode **children = CALLOC(new_children, tnode *);
  double *weights = CALLOC(new_children, double);

  int k = 0;
  tnode *curr = node->r_child;
  while(curr != last_existing_child) {
    weights[k] = curr->data->res.obj;
    children[k++] = curr;
    curr = curr->l_sib;
  }

  tnode *child = random_sample_weighted(children, weights, new_children);

  curr = node->r_child;
  while(curr != last_existing_child) {
    tnode *next = curr->l_sib;
    if(curr != child) {
      tree_remove_node(curr);
      free(free_subtree(curr));
    } 
    curr = next;
  }
  free(children);
  free(weights);

  return child;
}

double best_mcts_obj = -1e101;

mcts_stats mcts_simulate(env_t *env, tnode *start) {
  //randomly traverses the tree, starting at start and returns some statistics about the traversal

  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  tnode *root = CALLOC(1, tnode);
  memcpy(root, start, sizeof(tnode));
  root->l_sib = root->r_sib = root->l_child = root->r_child = NULL;

  tnode *curr = root;
  double best_max_obj = curr->data->res.max_poss_obj;
  //  hplane_data *best_max_vd = dup_hplane_data(curr->data);
  gsl_vector *best_max_v = gsl_vector_alloc(d);
  double best_obj = curr->data->res.obj;
  //  hplane_data *best_vd = dup_hplane_data(curr->data);
  gsl_vector *best_v = gsl_vector_alloc(d);
  while(curr->depth < d - 1) {
    //tnode *child = create_random_child(env, curr);
    tnode *child = create_random_child_weighted(env, curr);
    if(child->data->res.max_poss_obj >= best_max_obj) {
      best_max_obj = child->data->res.max_poss_obj;
      gsl_vector_memcpy(best_max_v, child->data->v);
    }
    if(child->data->res.obj >= best_obj) {
      best_obj = child->data->res.obj;
      gsl_vector_memcpy(best_v, child->data->v);
    }
    curr = child;
  }

  if(root->l_child)
    free(free_subtree(root->l_child));
  free(root);

  if(best_obj > best_mcts_obj) {
    printf("[MCTS] new best objective %g\n", best_obj);
    best_mcts_obj = best_obj;
  }

  return (mcts_stats) {
    .best_obj = best_obj,
    .best_vd = make_hplane_data(best_v, classify(env, best_v)),
    .best_max_obj = best_max_obj,
    .best_max_vd = make_hplane_data(best_max_v, classify(env, best_max_v))
  };
}

void mcts_expand(env_t *env, tnode *n, mcts_stats stats) {
  //TODO
  //need a better expansion stategy, and to intialize mean_obj
  if(!n)
    return;
  //create_node_children_notrim(env, n);
  //tnode *child = create_random_child(env, n);
  tnode *child = create_best_child1(env, n); //This is broken
  //tnode *child = create_random_child_weighted(env, n);
  if(child) {
    child->stats.mean_obj = child->data->res.obj;
  } else {
    mcts_expand(env, n->parent, stats);
  }
}

void mcts_backpropogate(env_t *env, tnode *n, mcts_stats stats) {
  //propogates up the tree, updating the averages and times_visiteds
  tnode *curr = n;
  while(curr) {
    int N = curr->stats.times_visited;
    double xbar = curr->stats.mean_obj;
    double obj = stats.best_obj;
    curr->stats.mean_obj = (xbar * N + obj)/(N+1);
    curr->stats.times_visited++;

    curr = curr->parent;
  }
}

hplane_list *mcts(env_t *env, gsl_vector *v0, hplane_list *A) {
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  class_res root_res = classify(env, v0);
  tnode root = (tnode) {make_hplane_data(v0, root_res), .branching_data=blank_branching_data(vs), .key=0}; //set vector v0 and depth = 0, nothing else

  double best_obj = -1e101;
  gsl_vector *best_v = v0;
  hplane_list *B = create_list();
  list_add_copy(B, root.data);

  time_t start = time(0);
  time_t dt = start;

  int nodes_explored = 0;

  while(dt - start < 120) {
    tnode *curr = mcts_select(env, &root);
    if(!curr) {
      printf("failed to select node\n");
      break;
    }
    nodes_explored++;
    if(time(0) - dt > 0) {
      printf("%d nodes explored, tree has %d nodes, best obj = %g, time diff = %ld\n", nodes_explored, tree_nodes, best_obj, dt - start);
      dt = time(0);
    }

    mcts_stats stats = mcts_simulate(env, curr);
    if(stats.best_vd) {
      gr_update_b(B, stats.best_vd);
    }
    if(stats.best_max_vd)
      gr_update_b(B, stats.best_max_vd);

    mcts_expand(env, curr, stats);

    mcts_backpropogate(env, curr, stats);

    if(curr->data->res.obj > best_obj) {
      best_obj = curr->data->res.obj;
      best_v = curr->data->v;
      printf("new best soln with objective %g\n", best_obj);
    }


    if(curr->depth < d - 1) {
      if(curr->key != 0)
	gr_update_b(B, curr->data); //no need to add the root twice
    } else {
      update_b_flips(env, curr, B);
    }
  }

  free_subtree(&root);
  gr_displace(env, B, A);
  free(B);
  return A;

}

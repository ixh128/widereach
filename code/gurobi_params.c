#include "helper.h"
#include "widereach.h"

gurobi_params_t *gurobi_params_default() {
  gurobi_params_t *p = CALLOC(1, gurobi_params_t);

  p->threads = 0;
  p->MIPFocus = 0;
  p->ImproveStartGap = 0;
  p->ImproveStartTime = GRB_INFINITY;
  p->VarBranch = 0;
  p->Heuristics = 0.05;
  p->Cuts = -1;
  p->RINS = -1;
  p->method = 0;
  p->init = NULL;
  p->pos_prio = 0;
  p->force_pos = 0;
  p->unbiased = 0;
  p->penalty = LAGRANGIAN;
  p->tm_lim = 120000;
  p->tm_lim_tune = 1200;
  return p;
}

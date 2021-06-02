
/**
 * a LFU module written in C++
 * note that the miss ratio of LFU cpp can be different from LFUFast
 * this is because of the difference in handling objects of same frequency
 *
 */

#include "libCacheSim/evictionAlgo/LFU.h"
#include "hashtable.h"
#include <cassert>

#include "abstractRank.h"


namespace eviction {
class LFU : public abstractRank {
public:
  LFU() = default;;
};
}



cache_t *LFU_init(common_cache_params_t ccache_params, void *init_params) {
  cache_t *cache = cache_struct_init("LFU", ccache_params);
  auto *lfu = new eviction::LFU;
  cache->eviction_params = lfu;

  cache->cache_init = LFU_init;
  cache->cache_free = LFU_free;
  cache->get = LFU_get;
  cache->check = LFU_check;
  cache->insert = LFU_insert;
  cache->evict = LFU_evict;
  cache->remove = LFU_remove;

  return cache;
}

void LFU_free(cache_t *cache) {
  auto *lfu = static_cast<eviction::LFU *>(cache->eviction_params);
  delete lfu;
  cache_struct_free(cache);
}

cache_ck_res_e LFU_check(cache_t *cache, request_t *req, bool update_cache) {
  auto *lfu = static_cast<eviction::LFU *>(cache->eviction_params);
  cache_obj_t *obj;
  auto res = cache_check_base(cache, req, update_cache, &obj);
  if (obj != nullptr && update_cache) {
    obj->rank.freq ++;
    obj->rank.last_access_vtime = (int64_t)req->n_req;
    auto itr = lfu->itr_map[obj];
    lfu->pq.erase(itr);
    itr = lfu->pq.emplace(obj, (double) obj->rank.freq, cache->vtime).first;
    lfu->itr_map[obj] = itr;
  }

  return res;
}

void LFU_insert(cache_t *cache, request_t *req) {
  auto *lfu = static_cast<eviction::LFU *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  obj->rank.freq = 1;
  obj->rank.last_access_vtime = (int64_t)req->n_req;

  auto itr = lfu->pq.emplace_hint(lfu->pq.begin(), obj, 1.0, cache->vtime);
  lfu->itr_map[obj] = itr;
  DEBUG_ASSERT(lfu->itr_map.size() == cache->n_obj);
}

void LFU_evict(cache_t *cache, request_t *req, cache_obj_t *evicted_obj) {
  auto *lfu = static_cast<eviction::LFU *>(cache->eviction_params);
  eviction::pq_node_type p = lfu->pick_lowest_score();
  cache_obj_t *obj = get<0>(p);
  if (evicted_obj != nullptr)
    memcpy(evicted_obj, obj, sizeof(cache_obj_t));

  cache_remove_obj_base(cache, obj);
}

cache_ck_res_e LFU_get(cache_t *cache, request_t *req) {
  return cache_get_base(cache, req);
}

void LFU_remove_obj(cache_t *cache, cache_obj_t *obj) {
  auto *lfu = static_cast<eviction::LFU *>(cache->eviction_params);
  lfu->remove_obj(cache, obj);
}


void LFU_remove(cache_t *cache, obj_id_t obj_id) {
  auto *lfu = static_cast<eviction::LFU *>(cache->eviction_params);
  lfu->remove(cache, obj_id);
}



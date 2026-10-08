import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import os
import json
#import sys 可能会用于调整
from openai import OpenAI
from tqdm import trange
from zai import ZhipuAiClient

from scenegraphnav.parser import parse_args
from gsamllavanav.evaluate import eval_planning_metrics
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from scenegraphnav.evaluate import run_episodes_batch
from scenegraphnav.agent import GeonavAgent, ChatAgent

DEVICE = 'cuda'
test_data = 'all'
zhipu_api_key = os.environ.get("GEONAV_ZHIPU_API_KEY", "")
#火山引擎豆包seed、deepseek
args = parse_args()

# 假设 arg.output_dir 是传入的参数
output_dir = args.output_dir
# 如果文件夹不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.mode == 'eval':
    objects = get_city_refer_objects()
    # 生成 episodes：如果指定了具体案例，则生成所有然后过滤；否则根据 test_one_example 生成
    if args.map_name is not None and args.episode_id is not None:
        # 指定了具体案例，生成所有 episodes 然后过滤
        test_episodes = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(args.split, test_data, args.altitude),
            max_episodes=None
        )
        test_episodes = [ep for ep in test_episodes if ep.map_name == args.map_name and ep.target_object.id == args.episode_id and (args.ann_id is None or ep.description_id == args.ann_id)]
        if not test_episodes:
            print(f"No episode found for map_name={args.map_name}, episode_id={args.episode_id}, ann_id={args.ann_id}")
            exit(1)
        print(f"Selected specific episode: map_name={args.map_name}, episode_id={args.episode_id}, ann_id={args.ann_id}")
    elif args.test_one_example:
        # 测试单个样例（随机或第一个）
        test_episodes = generate_episodes_from_mturk_trajectories(
            objects, 
            load_mturk_trajectories(args.split, 'all', args.altitude),
            max_episodes=1
        )
        print(f"Testing one example: map_name={test_episodes[0].map_name}, episode_id={test_episodes[0].id}")
    else:
        # 测试整个split的所有样例
        test_episodes = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(args.split, test_data, args.altitude),
            max_episodes=None
        )
    # 选择规划器运动至 landmark质点位置
    if args.landmark_mode == 'planner':
        # 使用run_episodes_batch处理landmark_mode为'planner'的情况
        trajectory_logs, target_xys = run_episodes_batch(args, None, test_episodes, DEVICE, landmark_mode='planner')
        # 计算指标并保存结果，然后结束程序（避免重复运行 agent）
        # metrics = eval_planning_metrics(args, test_episodes, trajectory_logs)
        # print(f"{args.split} -- NE {metrics.mean_final_pos_to_goal_dist: .1f}, SR {metrics.success_rate_final_pos_to_goal*100: .2f}, OSR {metrics.success_rate_oracle_pos_to_goal*100: .2f}, SPL {metrics.success_rate_weighted_by_path_length*100: .2f}metrics")
        # with open(args.output_dir + f'geonav_{args.split}_{args.ablation}_{test_data}.json', 'w') as f:
        #     json.dump({
        #         'metrics': metrics.to_dict(),
        #         'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
        #     }, f)
        # sys.exit(0)
    else:
        from gsamllavanav.observation import cropclient
        cropclient.load_image_cache()
        trajectory_logs = dict()
    
    # 初始化agents列表
    agents = []
    results = []
    
    def initialize_models():
        """Build configurable OpenAI-compatible VLM and LLM clients.

        API credentials are supplied through environment variables so this
        public reproduction checkout contains no private tokens.
        """
        local_base = os.environ.get("GEONAV_LOCAL_BASE_URL", "http://localhost:8000/v1")
        default_base = local_base if args.deployment == 'local' else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        vlm_base = os.environ.get("GEONAV_VLM_BASE_URL", default_base)
        llm_base = os.environ.get("GEONAV_LLM_BASE_URL", os.environ.get("DEEPSEEK_BASE_URL", vlm_base))
        shared_key = (
            os.environ.get("GEONAV_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
        vlm_key = os.environ.get("GEONAV_VLM_API_KEY", shared_key)
        llm_key = os.environ.get("GEONAV_LLM_API_KEY", shared_key)

        def make_client(api_key, base_url):
            client = OpenAI(api_key=api_key, base_url=base_url)
            # Some compatible providers enable hidden reasoning by default.
            # This opt-in keeps their output in the format expected by GeoNav
            # while leaving the official Qwen path unchanged by default.
            if os.environ.get("GEONAV_THINKING", "").lower() == "disabled":
                completions = client.chat.completions
                original_create = completions.create

                def create_without_thinking(*call_args, **call_kwargs):
                    extra_body = dict(call_kwargs.get("extra_body") or {})
                    extra_body.setdefault("thinking", {"type": "disabled"})
                    call_kwargs["extra_body"] = extra_body
                    call_kwargs.setdefault("reasoning_effort", "none")
                    return original_create(*call_args, **call_kwargs)

                completions.create = create_without_thinking
            return client

        return make_client(vlm_key, vlm_base), make_client(llm_key, llm_base)
    
    # model configuration
    vlmodel, llmodel = initialize_models()

    strategy_distance_records = {
        'Start': [],
        'Navigate': [],
        'Search': [],
        'Locate': []
    }
    output_directory = os.path.join(args.output_dir, "geonav_origin", args.split)
    os.makedirs(output_directory, exist_ok=True)  # 自动创建目录（包括父目录）
    metrics_jsonl_path = os.path.join(output_directory, f"episode_metrics_{args.split}.jsonl")

    # The checked-in value 2385 was a one-off resume offset and silently
    # skipped most episodes. Reproduction runs must cover the whole split.
    start_idx = 0
    end_idx = None
    # 确保 end_idx 不越界
    if end_idx is not None:
        end_idx = min(end_idx, len(test_episodes))
    episode_items = list(enumerate(test_episodes[start_idx:end_idx], start=start_idx))
    results = [None] * len(episode_items)

    # Resume from valid GeoNavAgent trajectory files. This matters for costly
    # API-backed inference runs: a process can stop after agents have persisted
    # their trajectories but before the aggregate metric writer finishes.
    agent_output_dir = os.path.join(args.output_dir, "full")
    os.makedirs(agent_output_dir, exist_ok=True)
    resumed_ids = set()
    from gsamllavanav.space import Pose4D
    for slot, (i, episode) in enumerate(episode_items):
        prefix = f"GeonavAgent_{episode.id}_"
        candidates = sorted(
            (name for name in os.listdir(agent_output_dir)
             if name.startswith(prefix) and name.endswith(".json")),
            reverse=True,
        )
        for name in candidates:
            path = os.path.join(agent_output_dir, name)
            try:
                with open(path, "r") as f:
                    saved = json.load(f)
                steps = saved.get("steps") or []
                trajectory = [Pose4D(*step["pose"]) for step in steps]
                if not trajectory:
                    continue
                trajectory_logs[episode.id] = trajectory
                results[slot] = bool(saved.get("success", False))
                resumed_ids.add(episode.id)
                for strategy, distance in (saved.get("strategy_distances") or {}).items():
                    strategy_distance_records.setdefault(strategy, []).append(distance)
                break
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                print(f"Ignoring invalid saved trajectory {path}: {exc}")

    episode_items = [item for item in episode_items if item[1].id not in resumed_ids]
    print(f"Resume: loaded {len(resumed_ids)} valid trajectories; {len(episode_items)} episodes remain")

    def run_episode(index_episode):
        i, episode = index_episode
        agent = GeonavAgent(args, episode.start_pose, episode, vlmodel, llmodel, set_height=None)
        agent.set_target(episode.target_position)
        result, trajectory_log = agent.run()
        return i, episode, result, trajectory_log, dict(agent.strategy_distances), dict(agent.strategy_timesteps)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    workers = max(1, int(args.num_workers))
    failed_episodes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_episode, item): item for item in episode_items}
        for future in as_completed(futures):
            i, episode = futures[future]
            try:
                i, episode, result, trajectory_log, episode_distances, episode_timesteps = future.result()
                if not trajectory_log:
                    raise ValueError("agent returned an empty trajectory")
                trajectory_logs[episode.id] = trajectory_log
                results[i - start_idx] = result
                for strategy, distance in episode_distances.items():
                    strategy_distance_records.setdefault(strategy, []).append(distance)
                print(f"Completed episode {i + 1}/{len(test_episodes)} ({len(trajectory_logs)} done)")
            except Exception as exc:
                failed_episodes.append({"index": i, "episode_id": str(episode.id), "error": repr(exc)})
                print(f"FAILED episode {i + 1}/{len(test_episodes)} ({episode.id}): {exc!r}")
                import traceback
                traceback.print_exc()

    # Write a deterministic, complete per-episode metrics file from the
    # recovered/generated trajectories. Never leave a misleading partial log.
    present_episodes = [episode for episode in test_episodes if episode.id in trajectory_logs]
    metric_rows = []
    for i, episode in enumerate(test_episodes):
        trajectory = trajectory_logs.get(episode.id)
        if not trajectory:
            continue
        episode_metrics = eval_planning_metrics(args, [episode], {episode.id: trajectory})
        metric_rows.append({
            "Episode": i + 1,
            "episode_id": str(episode.id),
            "NE": episode_metrics.mean_final_pos_to_goal_dist,
            "SR": episode_metrics.success_rate_final_pos_to_goal,
            "OSR": episode_metrics.success_rate_oracle_pos_to_goal,
            "SPL": episode_metrics.success_rate_weighted_by_path_length,
        })
    with open(metrics_jsonl_path, 'w') as metrics_f:
        for row in metric_rows:
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(present_episodes) != len(test_episodes):
        missing_ids = [str(e.id) for e in test_episodes if e.id not in trajectory_logs]
        failure_path = os.path.join(args.output_dir, f"incomplete_{args.split}.json")
        with open(failure_path, "w") as f:
            json.dump({"expected": len(test_episodes), "completed": len(present_episodes),
                       "failed": failed_episodes, "missing_episode_ids": missing_ids}, f, indent=2)
        raise RuntimeError(f"Incomplete evaluation: {len(present_episodes)}/{len(test_episodes)} trajectories; details: {failure_path}")

    metrics = eval_planning_metrics(args, test_episodes, trajectory_logs)

    print(f"{args.split} -- NE {metrics.mean_final_pos_to_goal_dist: .1f}, SR {metrics.success_rate_final_pos_to_goal*100: .2f}, OSR {metrics.success_rate_oracle_pos_to_goal*100: .2f}, SPL {metrics.success_rate_weighted_by_path_length*100: .2f}metrics")
    for strategy, distances in strategy_distance_records.items():
        if distances:  # 确保列表不为空
            avg_distance = sum(distances) / len(distances)
            print(f"Average distance for {strategy} : {avg_distance:.2f} (meters)")
        else:
            print(f"No data available for {strategy}")
    noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
    map_type = f"_{args.map_type}" if args.map_type != 'topdown_map' else ""
    
    result_path = os.path.join(args.output_dir, f'geonav_{args.split}_{args.ablation}_{test_data}{map_type}.json')
    with open(result_path, 'w') as f:
        json.dump({
            'metrics': metrics.to_dict(),
            'success': [results[i] if results[i] is not None else bool(
                any(row.get("episode_id") == str(ep.id) and row.get("SR", 0) == 1
                    for row in metric_rows)) for i, ep in enumerate(test_episodes)],
            'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
            'strategy_averages': {k: (sum(v)/len(v) if len(v)>0 else None) for k,v in strategy_distance_records.items()}
        }, f)

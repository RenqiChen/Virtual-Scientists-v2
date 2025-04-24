from sci_platform_fast import Platform
from utils.scientist_utils import read_txt_files_as_dict
import os
import argparse
import asyncio
from configs import deploy_config
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiments")
    # parser.add_argument(
    #     "--skip-idea-generation",
    #     action="store_true",
    #     help="Skip idea generation and load existing ideas",
    # )
    # parser.add_argument(
    #     "--skip-novelty-check",
    #     action="store_true",
    #     help="Skip novelty check and use existing ideas",
    # )
    # checkpoint
    parser.add_argument(
        "--model_name",
        type=str,
        default=deploy_config.model_name,
        help="employed base model",
    )
    parser.add_argument(
        "--leader_mode",
        type=str,
        default=deploy_config.leader_mode,
        help="who is the leader",
    )
    parser.add_argument(
        "--checkpoint",
        type=bool,
        default=deploy_config.checkpoint,
        help="Whether use checkpoint",
    )
    # test_time
    parser.add_argument(
        "--test_time",
        type=str,
        default=deploy_config.test_time,
        help='save new file',
    )
    # load_time
    parser.add_argument(
        "--load_time",
        type=str,
        default=deploy_config.load_time,
        help="load old file",
    )
    # root direction
    parser.add_argument(
        "--root_dir",
        type=str,
        default=deploy_config.root_dir,
        help="Dataset location.",
    )
    # how many scientists
    parser.add_argument(
        "--agent_num",
        type=int,
        # default=2000,
        default=deploy_config.agent_num,
        help="How many scientist leaders.",
    )
    parser.add_argument(
        "--ips",
        type=list,
        # default=2000,
        default=deploy_config.ips,
        help="How many ips are used.",
    )
    parser.add_argument(
        "--port",
        type=list,
        # default=list(range(11434, 11458)),
        default=deploy_config.port,
        help="How many ports are used."
    )
    # how many runs
    parser.add_argument(
        "--runs",
        type=int,
        # default = 1
        default=deploy_config.runs,
        help="Calculate average on how many runs.",
    )
    # team limit
    parser.add_argument(
        "--team_limit",
        type=int,
        # default = 4
        default=deploy_config.team_limit,
        help="Max number of teams for a scientist.",
    )
    parser.add_argument(
        "--max_discuss_iteration",
        type=int,
        # default = 1
        default=deploy_config.max_discuss_iteration,
        help="Max discuss iteration.",
    )
    parser.add_argument(
        "--max_team_member",
        type=int,
        # default = 6
        default=deploy_config.max_team_member,
        help="Max team mamber of a team, actual team size is max_team_member.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        # default = 50
        default=deploy_config.epochs,
        help="Epochs.",
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    args.save_dir = f'team_info/{args.max_discuss_iteration}_itrs_{args.max_team_member}_members'
    args.log_dir = f'team_log/{args.max_discuss_iteration}_itrs_{args.max_team_member}_members'

    end =False
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    while end==False:
        print(f'{len(os.listdir(args.save_dir))} files are created...')
        platform_example = Platform(
            root_dir = args.root_dir,
            agent_num=args.agent_num,
            model_name=args.model_name,
            ips=args.ips,
            port=args.port,
            team_limit = args.team_limit,
            group_max_discuss_iteration = args.max_discuss_iteration,
            max_teammember = args.max_team_member-1,
            log_dir = args.log_dir,
            info_dir = args.save_dir,
            checkpoint= args.checkpoint,
            test_time = args.test_time,
            load_time = args.load_time,
            leader_mode=args.leader_mode,
        )
        asyncio.run(platform_example.running(args.epochs))
        # try:
        #     platform_example = Platform(
        #         team_limit = args.team_limit,
        #         group_max_discuss_iteration = args.max_discuss_iteration,
        #         max_teammember = args.max_team_member-1,
        #         log_dir = args.log_dir,
        #         info_dir = args.save_dir
        #     )
        #     platform_example.running(args.epochs)
        # except:
        #     pass
        break
        if len(os.listdir(args.save_dir)) >= args.team_limit*args.runs:
            end = True

# paper_folder_path = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/papers"  # 替换为实际的文件夹路径
# paper_dicts = read_txt_files_as_dict(paper_folder_path)

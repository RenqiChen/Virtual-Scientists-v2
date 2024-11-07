from sci_platform_fast import Platform
from utils.scientist_utils import read_txt_files_as_dict
import os
import argparse
import asyncio
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
    # how many scientists
    parser.add_argument(
        "--agent_num",
        type=int,
        default=3169,
        help="How many scientist leaders.",
    )
    parser.add_argument(
        "--port",
        type=list,
        default=[11434,11435,11436,11437,11438,11439,11440,11441,11442,11443,11444,11445,11446,11447,11448,11449,11450,11451,11452],
        help="How many ports are used"
    )
    # how many runs
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Calculate average on how many runs.",
    )
    # team limit
    parser.add_argument(
        "--team_limit",
        type=int,
        default=2,
        help="Max number of teams for a scientist.",
    )
    parser.add_argument(
        "--max_discuss_iteration",
        type=int,
        default=1,
        help="Max discuss iteration.",
    )
    parser.add_argument(
        "--max_team_member",
        type=int,
        default=5,
        help="Max team mamber of a team, actual team size is max_team_member.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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
            agent_num=args.agent_num,
            port=args.port,
            team_limit = args.team_limit,
            group_max_discuss_iteration = args.max_discuss_iteration,
            max_teammember = args.max_team_member-1,
            log_dir = args.log_dir,
            info_dir = args.save_dir
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
        # break
        if len(os.listdir(args.save_dir)) >= args.team_limit*args.runs:
            end = True

# paper_folder_path = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/papers"  # 替换为实际的文件夹路径
# paper_dicts = read_txt_files_as_dict(paper_folder_path)
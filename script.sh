# 3 sampul x 1 citra rahasia
declare -a cover_path=(
    [0] = "splash"
    # [1] = 
    # [2] = 
)

declare -a secret_path ={
    [0] = "M_Left_index_finger"
    # [1] =
    # [2] =
}


declare -a cover_size =(
    [0] = 128
    [1] = 256
    [2] = 384
    [3] = 512
)

declare -a secret_size =(
    [0] = 8
    [1] = 16
    [2] = 24
)

declare -a joblist

## all possibility

for cp in "${cover_path[@]}"; do
    for sp in "${secret_path[@]}"; do
        for cs in "${cover_size[@]}"; do
            for ss in "${secret_size}"; do
                joblist+=("python3 main.py --cpath $cp --spath $secret_path --csize $cs --ssize $ss")
            done
        done
    done
done


jobmax = 20
jobcount = 0


for job in "${joblist[@]}" ; do
    ((++jobcount))
    [[ "${jobcount}" -gt "${jobmax}" ]] && wait -n && ((--jobcount))   # if jobcnt > 20 => wait for a job to finish, decrement jobcnt, then continue with next line ...
    ($job) &                                  # kick off new job
done

wait

exit



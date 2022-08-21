find ./train_img/data -regextype posix-extended -regex ".*\.(png|jpg)" | sort  > ./train_img/data_list.txt

find ./train_img/gt -regextype posix-extended -regex ".*\.(png|jpg)" | sort > ./train_img/gt_list.txt

paste -d  " " ./train_img/gt_list.txt ./train_img/data_list.txt > raindrop.txt

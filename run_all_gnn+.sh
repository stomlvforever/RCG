mkdir -p posotion_task
# 自动赋予执行权限（虽然这一步通常只需要执行一次）
chmod +x "$0"

python main.py --task_type 'capacitance' --gpu 0 --task_level 'edge' --net_only 0 --task classification --neg_edge_ratio 0.5 --batch_size 64 --lr 0.00001 --act_fn 'relu' --dropout 0.5 --layer_norm 1 --batch_norm 0  --use_bn 0 --residual 0 --g_bn 0 --g_drop 0 --g_ffn 0 --model gps_attention --local_gnn_type CustomGatedGCN --attn_dropout 0.5 --global_model_type None --num_heads 3 | tee posotion_task/classification.CustomGatedGCN.log &
#python main.py --gpu 1 --task_level 'edge' --net_only 0 --task classification --neg_edge_ratio 0.5 --batch_size 64 --lr 0.00001 --act_fn 'relu' --dropout 0.5 --layer_norm 1 --batch_norm 0  --use_bn 0 --residual 1 --g_bn 1 --g_drop 1 --g_ffn 1 --model gps_attention --local_gnn_type CustomGatedGCN --attn_dropout 0.5 --global_model_type None --num_heads 3 | tee posotion_task/classification.CustomGatedGCN+.log &
#python main.py --gpu 2 --task_level 'edge' --net_only 0 --task classification --neg_edge_ratio 0.5 --batch_size 64 --lr 0.00001 --act_fn 'relu' --dropout 0.5 --layer_norm 1 --batch_norm 0  --use_bn 0 --residual 0 --g_bn 0 --g_drop 0 --g_ffn 0 --model gps_attention --local_gnn_type CustomGCNConv --attn_dropout 0.5 --global_model_type None --num_heads 3 | tee posotion_task/classification.CustomGCNConv.log &
#python main.py --gpu 3 --task_level 'edge' --net_only 0 --task classification --neg_edge_ratio 0.5 --batch_size 64 --lr 0.00001 --act_fn 'relu' --dropout 0.5 --layer_norm 1 --batch_norm 0  --use_bn 0 --residual 1 --g_bn 1 --g_drop 1 --g_ffn 1 --model gps_attention --local_gnn_type CustomGCNConv --attn_dropout 0.5 --global_model_type None --num_heads 3 | tee posotion_task/classification.CustomGCNConv+.log &
#python main.py --gpu 4 --task_level 'edge' --net_only 0 --task classification --neg_edge_ratio 0.5 --batch_size 64 --lr 0.00001 --act_fn 'relu' --dropout 0.5 --layer_norm 1 --batch_norm 0  --use_bn 0 --residual 0 --g_bn 0 --g_drop 0 --g_ffn 0 --model gps_attention --local_gnn_type CustomGINEConv --attn_dropout 0.5 --global_model_type None --num_heads 3 | tee posotion_task/classification.CustomGINEConv.log &
#python main.py --gpu 5 --task_level 'edge' --net_only 0 --task classification --neg_edge_ratio 0.5 --batch_size 64 --lr 0.00001 --act_fn 'relu' --dropout 0.5 --layer_norm 1 --batch_norm 0  --use_bn 0 --residual 1 --g_bn 1 --g_drop 1 --g_ffn 1 --model gps_attention --local_gnn_type CustomGINEConv --attn_dropout 0.5 --global_model_type None --num_heads 3 | tee posotion_task/classification.CustomGINEConv+.log &



wait

echo "All training processes completed."


import concurrent.futures
import time
import os
import 侧窗倾角
import 底盘覆盖度
import 后三角窗阶差
import 顶棚挠度_轿车
import 顶棚挠度_SUV
import 顶棚X向尺寸_轿车
import 顶棚X向尺寸_SUV
import 后风挡角度
import 后风挡实际作用角
import 后视镜到车身距离
import 后视镜后端角度
import 后视镜镜柄厚度
import 后视镜末端平行段角度
import 后视镜末端平行段长度
import 后视镜前端角度
import 后视镜X向尺寸
import 后视镜Y向尺寸
import 后视镜Z向尺寸
import 接近角
import 离去角
import 亮条上端与侧窗阶差
import 亮条下端与侧窗阶差
import 前保两侧切线
import 前风挡角度
import 前风挡上端R角
import 前风挡下端R角
import 前风挡与车顶棚连接面差_轿车
import 前风挡与车顶棚连接面差_SUV
import 前格栅
import 前轮腔前X尺寸
import 前轮腔后X向尺寸
import 三角饰盖Z向最小夹角
import 引擎盖角度
import 引擎盖前端弧度倾角
import 迎风面积
import 最小离地间隙
import A柱空间角_X轴
import A柱空间角_Y轴
import A柱空间角_Z轴
import A柱上端平行段
import A柱上端与侧窗夹角
import A柱上端与亮条阶差
import A柱上端与前挡夹角
import A柱上端与前风挡阶差
import A柱上端X向尺寸
import A柱上端Y向尺寸
import A柱下端平行段
import A柱下端与侧窗夹角
import A柱下端与亮条阶差
import A柱下端与前挡夹角
import A柱下端与前风挡阶差
import A柱下端X向尺寸
import A柱下端Y向尺寸

# 定义所有需要执行的任务
def get_all_tasks_SUV(file_path):
    """获取所有需要执行的任务列表"""
    
    tasks = [
        ("侧窗倾角", 侧窗倾角.calculate, file_path),
        # 可以在这里添加更多任务，比如：
        ("顶棚挠度", 顶棚挠度_SUV .calculate, file_path),
        ("顶棚X向尺寸", 顶棚X向尺寸_SUV.calculate, file_path),
        ("底盘平整度", 底盘覆盖度.calculate, file_path),
        ("后风挡角度", 后风挡角度.calculate, file_path),
        ("后风挡实际作用角", 后风挡实际作用角.calculate, file_path),
        ("后三角窗阶差", 后三角窗阶差.calculate, file_path),
        ("后视镜到车身距离", 后视镜到车身距离.calculate, file_path),
        ("后视镜后端角度", 后视镜后端角度.calculate, file_path),
        ("后视镜镜柄厚度", 后视镜镜柄厚度.calculate, file_path),
        ("后视镜末端平行段角度", 后视镜末端平行段角度.calculate, file_path),
        ("后视镜末端平行段长度", 后视镜末端平行段长度.calculate, file_path),
        ("后视镜前端角度", 后视镜前端角度.calculate, file_path),
        ("后视镜X向尺寸", 后视镜X向尺寸.calculate, file_path),
        ("后视镜Y向尺寸", 后视镜Y向尺寸.calculate, file_path),
        ("后视镜Z向尺寸", 后视镜Z向尺寸.calculate, file_path),
        ("接近角", 接近角.calculate, file_path),
        ("离去角", 离去角.calculate, file_path),
        ("亮条上端与侧窗阶差", 亮条上端与侧窗阶差.calculate, file_path),
        ("亮条下端与侧窗阶差", 亮条下端与侧窗阶差.calculate, file_path),
        ("前保两侧切线方向与前轮距离", 前保两侧切线.calculate, file_path),
        ("前风挡角度", 前风挡角度.calculate, file_path),
        ("前风挡上端R角", 前风挡上端R角.calculate, file_path),
        ("前风挡下端R角", 前风挡下端R角.calculate, file_path),
        ("前风挡与车顶棚连接面差", 前风挡与车顶棚连接面差_SUV.calculate, file_path),
        ("前格栅角度", 前格栅.calculate, file_path),
        ("前轮腔前X向尺寸", 前轮腔前X尺寸.calculate, file_path),
        ("前轮腔后X向尺寸", 前轮腔后X向尺寸.calculate, file_path),
        ("三角饰盖Z向最小夹角", 三角饰盖Z向最小夹角.calculate, file_path),
        ("引擎盖角度", 引擎盖角度.calculate, file_path),
        ("引擎盖前端弧度倾角", 引擎盖前端弧度倾角.calculate, file_path),
        ("迎风面积", 迎风面积.calculate, file_path),
        ("离地间隙", 最小离地间隙.calculate, file_path),
        ("A柱与X轴空间角度", A柱空间角_X轴.calculate, file_path),
        ("A柱与Y轴空间角度", A柱空间角_Y轴.calculate, file_path),
        ("A柱与Z轴空间角度", A柱空间角_Z轴.calculate, file_path),
        ("A柱上端平行段", A柱上端平行段.calculate, file_path),
        ("A柱上端与侧窗夹角", A柱上端与侧窗夹角.calculate, file_path),
        ("A柱上端与亮条阶差", A柱上端与亮条阶差.calculate, file_path),
        ("A柱上端与前风挡夹角", A柱上端与前挡夹角.calculate, file_path),
        ("A柱上端与前风挡阶差", A柱上端与前风挡阶差.calculate, file_path),
        ("A柱上端X向尺寸", A柱上端X向尺寸.calculate, file_path),
        ("A柱上端Y向尺寸", A柱上端Y向尺寸.calculate, file_path),
        ("A柱下端平行段", A柱下端平行段.calculate, file_path),
        ("A柱下端与侧窗夹角", A柱下端与侧窗夹角.calculate, file_path),
        ("A柱下端与亮条阶差", A柱下端与亮条阶差.calculate, file_path),
        ("A柱下端与前风挡夹角", A柱下端与前挡夹角.calculate, file_path),
        ("A柱下端与前风挡阶差", A柱下端与前风挡阶差.calculate, file_path),
        ("A柱下端X向尺寸", A柱下端X向尺寸.calculate, file_path),
        ("A柱下端Y向尺寸", A柱下端Y向尺寸.calculate, file_path),
        
    ]
    
    return tasks

def get_all_tasks_sedan(file_path):
    """获取所有需要执行的任务列表"""
    
    tasks = [
        ("侧窗倾角", 侧窗倾角.calculate, file_path),
        # 可以在这里添加更多任务，比如：
        ("顶棚挠度", 顶棚挠度_轿车.calculate, file_path),
        ("顶棚X向尺寸", 顶棚X向尺寸_轿车.calculate, file_path),
        ("底盘平整度", 底盘覆盖度.calculate, file_path),
        ("后风挡角度", 后风挡角度.calculate, file_path),
        ("后风挡实际作用角", 后风挡实际作用角.calculate, file_path),
        ("后三角窗阶差", 后三角窗阶差.calculate, file_path),
        ("后视镜到车身距离", 后视镜到车身距离.calculate, file_path),
        ("后视镜后端角度", 后视镜后端角度.calculate, file_path),
        ("后视镜镜柄厚度", 后视镜镜柄厚度.calculate, file_path),
        ("后视镜末端平行段角度", 后视镜末端平行段角度.calculate, file_path),
        ("后视镜末端平行段长度", 后视镜末端平行段长度.calculate, file_path),
        ("后视镜前端角度", 后视镜前端角度.calculate, file_path),
        ("后视镜X向尺寸", 后视镜X向尺寸.calculate, file_path),
        ("后视镜Y向尺寸", 后视镜Y向尺寸.calculate, file_path),
        ("后视镜Z向尺寸", 后视镜Z向尺寸.calculate, file_path),
        ("接近角", 接近角.calculate, file_path),
        ("离去角", 离去角.calculate, file_path),
        ("亮条上端与侧窗阶差", 亮条上端与侧窗阶差.calculate, file_path),
        ("亮条下端与侧窗阶差", 亮条下端与侧窗阶差.calculate, file_path),
        ("前保两侧切线方向与前轮距离", 前保两侧切线.calculate, file_path),
        ("前风挡角度", 前风挡角度.calculate, file_path),
        ("前风挡上端R角", 前风挡上端R角.calculate, file_path),
        ("前风挡下端R角", 前风挡下端R角.calculate, file_path),
        ("前风挡与车顶棚连接面差", 前风挡与车顶棚连接面差_轿车.calculate, file_path),
        ("前格栅角度", 前格栅.calculate, file_path),
        ("前轮腔前X向尺寸", 前轮腔前X尺寸.calculate, file_path),
        ("前轮腔后X向尺寸", 前轮腔后X向尺寸.calculate, file_path),
        ("三角饰盖Z向最小夹角", 三角饰盖Z向最小夹角.calculate, file_path),
        ("引擎盖角度", 引擎盖角度.calculate, file_path),
        ("引擎盖前端弧度倾角", 引擎盖前端弧度倾角.calculate, file_path),
        ("迎风面积", 迎风面积.calculate, file_path),
        ("离地间隙", 最小离地间隙.calculate, file_path),
        ("A柱与X轴空间角度", A柱空间角_X轴.calculate, file_path),
        ("A柱与Y轴空间角度", A柱空间角_Y轴.calculate, file_path),
        ("A柱与Z轴空间角度", A柱空间角_Z轴.calculate, file_path),
        ("A柱上端平行段", A柱上端平行段.calculate, file_path),
        ("A柱上端与侧窗夹角", A柱上端与侧窗夹角.calculate, file_path),
        ("A柱上端与亮条阶差", A柱上端与亮条阶差.calculate, file_path),
        ("A柱上端与前风挡夹角", A柱上端与前挡夹角.calculate, file_path),
        ("A柱上端与前风挡阶差", A柱上端与前风挡阶差.calculate, file_path),
        ("A柱上端X向尺寸", A柱上端X向尺寸.calculate, file_path),
        ("A柱上端Y向尺寸", A柱上端Y向尺寸.calculate, file_path),
        ("A柱下端平行段", A柱下端平行段.calculate, file_path),
        ("A柱下端与侧窗夹角", A柱下端与侧窗夹角.calculate, file_path),
        ("A柱下端与亮条阶差", A柱下端与亮条阶差.calculate, file_path),
        ("A柱下端与前风挡夹角", A柱下端与前挡夹角.calculate, file_path),
        ("A柱下端与前风挡阶差", A柱下端与前风挡阶差.calculate, file_path),
        ("A柱下端X向尺寸", A柱下端X向尺寸.calculate, file_path),
        ("A柱下端Y向尺寸", A柱下端Y向尺寸.calculate, file_path),
        
    ]
    
    return tasks

def run_tasks_with_threadpool(file_path, vehicle_type):
    """使用线程池执行所有任务"""
    # 获取所有任务
    if vehicle_type == "轿车":
        tasks = get_all_tasks_sedan(file_path)
    elif vehicle_type == "SUV":
        tasks = get_all_tasks_SUV(file_path)
    
    # 获取CPU核心数，限制最大线程数
    cpu_count = os.cpu_count() or 4
    max_workers = min(len(tasks), cpu_count)  # I/O密集型任务可以设置为核心数的2倍
    
    print(f"总共有 {len(tasks)} 个任务，使用 {max_workers} 个线程")
    
    results = {}
    
    start_time = time.time()
    
    # 使用线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {}
        for task_name, func, *args in tasks:
            future = executor.submit(func, *args)
            future_to_task[future] = task_name
        
        # 获取所有结果
        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                results[task_name] = result
                #print(f"✓ {task_name} 完成: {result}")
            except Exception as e:
                #print(f"✗ {task_name} 执行出错: {e}")
                results[task_name] = None
    
    #将结果根据界面展示顺序进行排序
    car_parameters = [
    "A柱上端X向尺寸",
    "A柱上端Y向尺寸",
    "A柱上端与前风挡阶差",
    "A柱上端与亮条阶差",
    "亮条上端与侧窗阶差",
    "A柱上端与侧窗夹角",
    "A柱上端与前风挡夹角",
    "A柱上端平行段",
    "前风挡上端R角",
    "A柱下端X向尺寸",
    "A柱下端Y向尺寸",
    "A柱下端与前风挡阶差",
    "A柱下端与亮条阶差",
    "亮条下端与侧窗阶差",
    "A柱下端与侧窗夹角",
    "A柱下端与前风挡夹角",
    "A柱下端平行段",
    "前风挡下端R角",
    "后视镜X向尺寸",
    "后视镜Y向尺寸",
    "后视镜Z向尺寸",
    "三角饰盖Z向最小夹角",
    "后视镜前端角度",
    "后视镜后端角度",
    "后视镜到车身距离",
    "后视镜末端平行段角度",
    "后视镜末端平行段长度",
    "顶棚挠度",
    "顶棚X向尺寸",
    "后风挡角度",
    "后风挡实际作用角",
    "前风挡角度",
    "前风挡与车顶棚连接面差",
    "引擎盖角度",
    "引擎盖前端弧度倾角",
    "前格栅角度",
    "接近角",
    "离去角",
    "前轮腔前X向尺寸",
    "前轮腔后X向尺寸",
    "前保两侧切线方向与前轮距离",
    "迎风面积",
    "侧窗倾角",
    "后视镜镜柄厚度",
    "A柱与X轴空间角度",
    "A柱与Y轴空间角度",
    "A柱与Z轴空间角度",
    "离地间隙",
    "后三角窗阶差",
    "底盘平整度"]
    
    # 统计总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    #结果排序
    order_results = [round(results[key], 3) if results[key] is not None else None 
                        for key in car_parameters if key in results]
    return order_results

if __name__ == "__main__":
    start_time = time.time()
    
    # 执行所有任务
    file_path = r"E:\一汽风噪\点云文件\AITO M9 Outer Surface.stl"
    vehicle_type = "轿车"
    order_results = run_tasks_with_threadpool(file_path, vehicle_type)
    
    
    print(f"\n=== 执行完成 ===")
    #print(f"总耗时: {total_time:.2f} 秒")
    print(f"完成任务数: {len([r for r in order_results if r is not None])}")
    print(f"失败任务数: {len([r for r in order_results if r is None])}")
    
    # 输出所有结果
    # print(f"\n=== 详细结果 ===")
    # for task_name, result in all_results.items():
    #     print(f"{task_name}: {result}")
        
    print(len(order_results))

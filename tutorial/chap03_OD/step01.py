#!/usr/bin/env python3
"""
다양한 데이터셋 구조를 자동 감지하여 train/validation으로 분할하는 스크립트
지원하는 구조:
1. dataset/images/train, dataset/images/val
2. dataset/train, dataset/val  
3. dataset/images/ (단일 폴더)
4. dataset/ (이미지 파일들이 직접 있는 경우)
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

def get_image_files(folder_path: str) -> List[str]:
    """폴더에서 이미지 파일들을 가져옵니다."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    folder_path = Path(folder_path)
    image_files = []
    
    if not folder_path.exists():
        return image_files
    
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files

def detect_dataset_structure(dataset_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """데이터셋 구조를 자동으로 감지합니다."""
    dataset_path = Path(dataset_path)
    
    # 구조 1: dataset/images/train, dataset/images/val
    images_train = dataset_path / 'images' / 'train'
    images_val = dataset_path / 'images' / 'val'
    if images_train.exists() or images_val.exists():
        return 'images_trainval', str(images_train), str(images_val)
    
    # 구조 2: dataset/train, dataset/val
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'
    if train_dir.exists() or val_dir.exists():
        return 'trainval', str(train_dir), str(val_dir)
    
    # 구조 3: dataset/images/ (단일 폴더)
    images_dir = dataset_path / 'images'
    if images_dir.exists() and get_image_files(images_dir):
        return 'images_only', str(images_dir), None
    
    # 구조 4: dataset/ (이미지 파일들이 직접 있는 경우)
    if get_image_files(dataset_path):
        return 'direct', str(dataset_path), None
    
    return None, None, None

def split_files(files: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    """파일 리스트를 train/validation으로 분할합니다."""
    random.shuffle(files)
    val_count = int(len(files) * val_ratio)
    
    val_files = files[:val_count]
    train_files = files[val_count:]
    
    return train_files, val_files

def process_files(file_paths: List[str], dst_folder: str, move: bool = True):
    """파일들을 대상 폴더로 이동 또는 복사합니다."""
    dst_path = Path(dst_folder)
    
    # 대상 폴더 생성 (부모 폴더까지 모두 생성)
    try:
        dst_path.mkdir(parents=True, exist_ok=True)
        print(f"  📁 대상 폴더 생성: {dst_path}")
    except Exception as e:
        print(f"❌ 폴더 생성 실패: {dst_path} -> {e}")
        return 0
    
    processed_count = 0
    action = "이동" if move else "복사"
    
    print(f"  📋 처리할 파일 수: {len(file_paths)}개")
    
    for i, file_path in enumerate(file_paths):
        src_file = Path(file_path)
        dst_file = dst_path / src_file.name
        
        if src_file.exists():
            try:
                if move:
                    shutil.move(str(src_file), str(dst_file))
                else:
                    shutil.copy2(src_file, dst_file)
                processed_count += 1
                
                # 처리 진행률 표시 (10개마다)
                if (i + 1) % 10 == 0 or (i + 1) == len(file_paths):
                    print(f"  ✅ {action} 완료: {processed_count}/{len(file_paths)}개")
                    
            except Exception as e:
                print(f"⚠️  {action} 실패: {src_file} -> {e}")
        else:
            print(f"⚠️  파일을 찾을 수 없습니다: {src_file}")
    
    print(f"  🎯 최종 {action} 완료: {processed_count}개")
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='다양한 구조의 데이터셋을 train/validation으로 분할')
    parser.add_argument('dataset_path', help='데이터셋 경로')
    parser.add_argument('--val-ratio', type=float, default=0.2, 
                       help='validation 비율 (기본값: 0.2)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--copy', action='store_true', 
                       help='파일을 복사합니다 (기본값: 이동)')
    parser.add_argument('--move', action='store_true', 
                       help='파일을 이동합니다 (기본 동작)')
    
    args = parser.parse_args()
    
    # 이동/복사 모드 결정 (기본값: 이동)
    if args.copy and args.move:
        print("❌ --copy와 --move 옵션을 동시에 사용할 수 없습니다.")
        return
    
    move_mode = not args.copy  # copy가 지정되지 않으면 이동 모드
    
    # 랜덤 시드 설정
    random.seed(args.seed)
    # 구조별 출력 경로 설정
    if structure_type == 'images_trainval':
        # dataset/images/train, dataset/images/val -> output/images/train, output/images/val
        output_train_path = output_path / 'images' / 'train'
        output_val_path = output_path / 'images' / 'val'
    elif structure_type == 'trainval':
        # dataset/train, dataset/val -> output/train, output/val
        output_train_path = output_path / 'train'
        output_val_path = output_path / 'val'
    else:
        # 단일 폴더인 경우 images/train, images/val 구조로 출력
        output_train_path = output_path / 'images' / 'train'
        output_val_path = output_path / 'images' / 'val'
    
    # 데이터셋 경로 확인
    if not dataset_path.exists():
        print(f"❌ 데이터셋 경로가 존재하지 않습니다: {dataset_path}")
        return
    
    # 데이터셋 구조 자동 감지
    print(f"🔍 데이터셋 구조 감지 중: {dataset_path}")
    structure_type, train_path, val_path = detect_dataset_structure(args.dataset_path)
    
    if structure_type is None:
        print(f"❌ 지원하는 데이터셋 구조를 찾을 수 없습니다.")
        print(f"지원하는 구조:")
        print(f"  1. {dataset_path}/images/train, {dataset_path}/images/val")
        print(f"  2. {dataset_path}/train, {dataset_path}/val")
        print(f"  3. {dataset_path}/images/ (이미지 파일들)")
        print(f"  4. {dataset_path}/ (이미지 파일들)")
        return
    
    # 출력 폴더 생성
    try:
        output_train_path.mkdir(parents=True, exist_ok=True)
        output_val_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 출력 폴더 확인/생성: {output_train_path.parent}")
    except Exception as e:
        print(f"❌ 출력 폴더 생성 실패: {output_train_path.parent} -> {e}")
        return
    
    # 구조별 처리
    all_files = []
    
    if structure_type == 'images_trainval':
        print(f"📁 감지된 구조: images/train, images/val")
        train_files = get_image_files(train_path) if Path(train_path).exists() else []
        val_files = get_image_files(val_path) if Path(val_path).exists() else []
        all_files = train_files + val_files
        print(f"🖼️  기존 Train: {len(train_files)}개, Val: {len(val_files)}개")
        
    elif structure_type == 'trainval':
        print(f"📁 감지된 구조: train, val")
        train_files = get_image_files(train_path) if Path(train_path).exists() else []
        val_files = get_image_files(val_path) if Path(val_path).exists() else []
        all_files = train_files + val_files
        print(f"🖼️  기존 Train: {len(train_files)}개, Val: {len(val_files)}개")
        
    elif structure_type == 'images_only':
        print(f"📁 감지된 구조: images/ (단일 폴더)")
        all_files = get_image_files(train_path)  # train_path가 images 폴더 경로
        print(f"🖼️  전체 이미지: {len(all_files)}개")
        
    elif structure_type == 'direct':
        print(f"📁 감지된 구조: 직접 이미지 파일들")
        all_files = get_image_files(train_path)  # train_path가 dataset 폴더 경로
        print(f"🖼️  전체 이미지: {len(all_files)}개")
    
    if not all_files:
        print(f"❌ 이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"🖼️  총 이미지: {len(all_files)}개")
    print(f"📊 새로운 Validation 비율: {args.val_ratio:.1%}")
    print(f"📤 출력 경로: {dataset_path}/images/")
    print(f"🔄 모드: {'이동' if move_mode else '복사'}")
    print("-" * 60)
    
    # 새로운 비율로 분할
    new_train_files, new_val_files = split_files(all_files, args.val_ratio)
    
    print(f"🚂 새로운 Train 이미지: {len(new_train_files)}개")
    print(f"✅ 새로운 Validation 이미지: {len(new_val_files)}개")
    print("-" * 60)
    
    # 파일 처리
    action_word = "이동" if move_mode else "복사"
    print(f"📋 Train 이미지 {action_word} 중...")
    train_processed = process_files(new_train_files, output_train_path, move_mode)
    
    print(f"📋 Validation 이미지 {action_word} 중...")
    val_processed = process_files(new_val_files, output_val_path, move_mode)
    
    print("-" * 60)
    print(f"🎉 재분할 완료!")
    print(f"📊 총 이미지: {len(all_files)}개")
    print(f"🚂 Train: {train_processed}개 {action_word} → {output_train_path}")
    print(f"✅ Validation: {val_processed}개 {action_word} → {output_val_path}")
    
    # 실제 비율 계산
    total_processed = train_processed + val_processed
    if total_processed > 0:
        actual_train_ratio = train_processed / total_processed
        actual_val_ratio = val_processed / total_processed
        print(f"📈 실제 비율 - Train: {actual_train_ratio:.1%}, Val: {actual_val_ratio:.1%}")
            
    if move_mode:
        print(f"⚠️  원본 파일들이 이동되었습니다. 백업이 필요한 경우 --copy 옵션을 사용하세요.")

if __name__ == "__main__":
    main()
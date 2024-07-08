import oss2
from get_sts_auth import get_sts_auth
from tqdm import tqdm

# yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
region='cn-shanghai'
endpoint = f'https://oss-{region}.aliyuncs.com'
# 填写Bucket名称，例如examplebucket。
bucket_name = 'wordshk-eg-pr-siri-female'
role_arn = 'acs:ram::1945506368897984:role/oss'

sts_auth = get_sts_auth(region, role_arn)

# 使用临时访问凭证中的认证信息初始化StsAuth实例。
auth = oss2.StsAuth(sts_auth['access_key_id'],
                    sts_auth['access_key_secret'],
                    sts_auth['security_token'])

bucket = oss2.Bucket(auth, endpoint, bucket_name)

# List and delete .m4a files
def clear_files():
    # List all files in the bucket
    files_to_delete = [obj.key for obj in oss2.ObjectIterator(bucket)]
    print(f'Total files to delete: {len(files_to_delete)}')
    
    # Delete all files with tqdm
    for file_key in tqdm(files_to_delete):
        result = bucket.delete_object(file_key)
        if result.status != 204:
            print(f'Failed to delete: {file_key}')

# Run the function to delete .m4a files
clear_files()

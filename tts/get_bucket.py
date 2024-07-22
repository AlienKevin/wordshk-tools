import oss2
from get_sts_auth import get_sts_auth

def get_bucket():
    # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    region='cn-shanghai'
    endpoint = f'https://oss-{region}.aliyuncs.com'
    # 填写Bucket名称，例如examplebucket。
    bucket_name = 'wordshk-pr-hiujin'
    role_arn = 'acs:ram::1945506368897984:role/oss'

    sts_auth = get_sts_auth(region, role_arn)

    # 使用临时访问凭证中的认证信息初始化StsAuth实例。
    auth = oss2.StsAuth(sts_auth['access_key_id'],
                        sts_auth['access_key_secret'],
                        sts_auth['security_token'])

    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    
    return bucket

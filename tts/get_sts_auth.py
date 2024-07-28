# -*- coding: utf-8 -*-

from aliyunsdkcore import client
from aliyunsdkcore.request import CommonRequest
import json
import oss2
import os

def get_sts_auth(region, role_arn):
    # 填写步骤1创建的RAM用户AccessKey。
    access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
    access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')

    # 创建权限策略。
    # 仅允许对examplebucket执行上传（PutObject）和下载（GetObject）操作。
    policy_text = '{"Version": "1", "Statement": [{"Action": ["oss:*"], "Effect": "Allow", "Resource": ["*"]}]}'

    clt = client.AcsClient(access_key_id, access_key_secret, region)
    request = CommonRequest(product="Sts", version='2015-04-01', action_name='AssumeRole')
    request.set_method('POST')
    request.set_protocol_type('https')
    request.add_query_param('RoleArn', role_arn)
    # 指定自定义角色会话名称，用来区分不同的令牌，例如填写为sessiontest。
    request.add_query_param('RoleSessionName', 'sessiontest')
    # 指定临时访问凭证有效时间单位为秒，最小值为900，最大值根据STS角色设置。
    request.add_query_param('DurationSeconds', '14400')
    # 如果policy为空，则RAM用户默认获得该角色下所有权限。如果有特殊权限控制要求，请参考上述policy_text设置。
    request.add_query_param('Policy', policy_text)
    request.set_accept_format('JSON')

    body = clt.do_action_with_exception(request)

    # 使用RAM用户的AccessKey ID和AccessKey Secret向STS申请临时访问凭证。
    token = json.loads(oss2.to_unicode(body))
    # 打印STS返回的临时访问密钥（AccessKey ID和AccessKey Secret）、安全令牌（SecurityToken）以及临时访问凭证过期时间（Expiration）。
    print('AccessKeyId: '+token['Credentials']['AccessKeyId'])
    print('AccessKeySecret: '+token['Credentials']['AccessKeySecret'])
    print('SecurityToken: '+token['Credentials']['SecurityToken'])
    print('Expiration: '+token['Credentials']['Expiration'])

    return {
        'access_key_id': token['Credentials']['AccessKeyId'],
        'access_key_secret': token['Credentials']['AccessKeySecret'],
        'security_token': token['Credentials']['SecurityToken'],
    }

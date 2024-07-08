# OSS

Follow tutoria: https://help.aliyun.com/zh/oss/developer-reference/authorized-access

Create an OSS role that has the following policy:
```json
{
    "Version": "1",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "oss:*",
            "Resource": "*"
        }
    ]
}
```

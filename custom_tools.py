import boto3

def list_public_buckets():
    s3 = boto3.client('s3')
    public_buckets = []

    for bucket in s3.list_buckets()['Buckets']:
        try:
            bucket_acl = s3.get_bucket_acl(Bucket=bucket['Name'])
            grants = bucket_acl['Grants']
            public_grant = next(
                (x for x in grants if x['Grantee'].get('URI') == 'http://acs.amazonaws.com/groups/global/AllUsers'),
                None
            )
            if public_grant:
                public_buckets.append(bucket['Name'])
        except Exception as e:
            print(f"Error: {e}")

    return public_buckets

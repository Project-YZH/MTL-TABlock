const express = require('express')
const app = express();
const bodyParser = require('body-parser');
const port = 3000;
const cors = require('cors');
const fs = require('fs');

const jsonfile = require('jsonfile');
let website = ['null'];

app.use(cors({
    credentials: true,
    origin: true
}));
app.use(bodyParser.urlencoded({
    limit: '100mb', extended: true
}));
app.use(bodyParser.json({ limit: '100mb' }));


async function insertRequest(newHttpReq, website) {
    const file = 'output/' + website + '/request.json';
    jsonfile.writeFile(file, newHttpReq, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}

async function insertRequestInfo(newHttpReq, website) {
    const file = 'output/' + website + '/requestInfo.json';
    jsonfile.writeFile(file, newHttpReq, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}

async function insertResponse(newHttpResp, website) {
    const file = 'output/' + website + '/responses.json';
    jsonfile.writeFile(file, newHttpResp, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}

async function saveResponse(request_id, body, website) {
    const file = 'output/' + website + '/response/'+ request_id + ".txt";
    fs.writeFile(file, body, (err) => {
        if (err) {
          console.log(err);
        }
      });
}

async function insertInfo(newInfo, website) {
    const file = 'output/' + website + '/cookie_storage.json';
    jsonfile.writeFile(file, newInfo, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}

async function debugInfo(newInfo, website) {
    const file = 'output/' + website + '/debug.json';
    jsonfile.writeFile(file, newInfo, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}

async function insertEventSet(newInfo, website) {
    const file = 'output/' + website + '/eventset.json';
    jsonfile.writeFile(file, newInfo, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}

async function insertEventGet(newInfo, website) {
    const file = 'output/' + website + '/eventget.json';
    jsonfile.writeFile(file, newInfo, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}

async function insertScriptIds(newInfo, website) {
    const file = 'output/' + website + '/script_ids.json';
    jsonfile.writeFile(file, newInfo, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}


app.post('/request', (req, res) => {
    if (req.body.http_req != `http://localhost:${port}/cookiestorage`) {
        req.body.top_level_url = website[0];
        insertRequest(req.body, website[0]);
    }
    res.send("request-success");
})

app.post('/requestinfo', (req, res) => {
    insertRequestInfo(req.body, website[0]);
    res.send("requestInfo-success");
})

app.post('/response', (req, res) => {
    insertResponse(req.body, website[0]);
    res.send("response-success");
})

app.post('/respfile', (req, res) => {
    saveResponse(req.body.request_id,req.body.response,website[0]);
    res.send("saveresponse-success");
})

app.post('/cookiestorage', (req, res) => {
    insertInfo(req.body, website[0]);
    res.send("cookie-success");
})

app.post('/debug', (req, res) => {
    debugInfo(req.body, website[0]);
    res.send("debug-success");
})

app.post('/eventset', (req, res) => {
    insertEventSet(req.body, website[0]);
    res.send("eventset-success");
})

app.post('/eventget', (req, res) => {
    insertEventGet(req.body, website[0]);
    res.send("eventget-success");
})

app.post('/scriptid', (req, res) => {
    insertScriptIds(req.body, website[0]);
    res.send("scriptids-success");
})

app.post('/complete', (req, res) => {
    website[0] = req.body.website;
    res.send("complete-success");
})


app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`);
})